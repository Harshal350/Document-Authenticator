"""
Marker detection engine for DocuGuard fake document detection.
Detects various forgery indicators across multiple categories.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default marker rules configuration
DEFAULT_MARKER_RULES: Dict[str, Any] = {
    "categories": {
        "dates": {
            "rules": [
                {
                    "name": "impossible_dates",
                    "description": "Dates that cannot exist (e.g., Feb 30, Apr 31)",
                    "severity": "high",
                    "score": 0.9,
                    "patterns": [
                        r"\b(?:02)[/-](?:30|31)[/-]\d{2,4}\b",
                        r"\b(?:04|06|09|11)[/-](?:31)[/-]\d{2,4}\b",
                        r"\b(?:02)[/-](?:29)[/-](?:(?:1[6-9]|[2-9]\d)(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)\b",
                    ],
                },
                {
                    "name": "future_dates",
                    "description": "Dates significantly in the future",
                    "severity": "medium",
                    "score": 0.6,
                    "patterns": [],
                },
                {
                    "name": "issue_after_expiry",
                    "description": "Issue date appears after expiry date",
                    "severity": "high",
                    "score": 0.85,
                    "patterns": [],
                },
                {
                    "name": "unusual_date_formats",
                    "description": "Inconsistent or unusual date formatting",
                    "severity": "low",
                    "score": 0.3,
                    "patterns": [],
                },
            ],
        },
        "numbers": {
            "rules": [
                {
                    "name": "inconsistent_totals",
                    "description": "Subtotal + tax does not equal total",
                    "severity": "high",
                    "score": 0.9,
                    "patterns": [],
                },
                {
                    "name": "amount_mismatches",
                    "description": "Same line item shows different amounts",
                    "severity": "high",
                    "score": 0.85,
                    "patterns": [],
                },
                {
                    "name": "percentage_inconsistencies",
                    "description": "Percentages do not add up to expected totals",
                    "severity": "medium",
                    "score": 0.7,
                    "patterns": [],
                },
                {
                    "name": "repeated_numbers",
                    "description": "Suspiciously repeated numerical values",
                    "severity": "medium",
                    "score": 0.5,
                    "patterns": [],
                },
            ],
        },
        "text": {
            "rules": [
                {
                    "name": "inconsistent_terminology",
                    "description": "Same concept referred to by different terms",
                    "severity": "medium",
                    "score": 0.6,
                    "patterns": [],
                },
                {
                    "name": "suspicious_phrases",
                    "description": "Phrases commonly found in forged documents",
                    "severity": "medium",
                    "score": 0.7,
                    "patterns": [
                        r"this\s+is\s+a\s+certified\s+copy",
                        r"certified\s+true\s+copy",
                        r"original\s+document",
                        r"for\s+illustrative\s+purposes",
                        r"sample\s+document",
                        r"specimen\s+copy",
                        r"draft\s+copy",
                    ],
                },
                {
                    "name": "repeated_sentences",
                    "description": "Exact sentences repeated multiple times",
                    "severity": "medium",
                    "score": 0.6,
                    "patterns": [],
                },
                {
                    "name": "abnormal_capitalization",
                    "description": "Unusual or excessive use of capitalization",
                    "severity": "low",
                    "score": 0.4,
                    "patterns": [],
                },
                {
                    "name": "excessive_punctuation",
                    "description": "Excessive use of punctuation marks",
                    "severity": "low",
                    "score": 0.35,
                    "patterns": [],
                },
                {
                    "name": "template_like_language",
                    "description": "Language patterns typical of template documents",
                    "severity": "medium",
                    "score": 0.55,
                    "patterns": [
                        r"\[.*?name.*?\]",
                        r"\[.*?date.*?\]",
                        r"\[.*?address.*?\]",
                        r"\bfill\s+in\s+(?:the\s+)?(?:below|following)\b",
                        r"\binsert\s+(?:your|the)\b",
                        r"\b(?:enter|provide)\s+(?:your|the)\b",
                    ],
                },
            ],
        },
        "formatting": {
            "rules": [
                {
                    "name": "inconsistent_font_sizes",
                    "description": "Unusually high variance in font sizes",
                    "severity": "medium",
                    "score": 0.5,
                    "patterns": [],
                },
                {
                    "name": "alignment_inconsistencies",
                    "description": "Multiple alignment styles in same document",
                    "severity": "low",
                    "score": 0.4,
                    "patterns": [],
                },
                {
                    "name": "missing_sections",
                    "description": "Expected sections are missing from document",
                    "severity": "medium",
                    "score": 0.6,
                    "patterns": [],
                },
            ],
        },
        "metadata": {
            "rules": [
                {
                    "name": "suspicious_timestamps",
                    "description": "Creation and modification dates too close or inconsistent",
                    "severity": "medium",
                    "score": 0.5,
                    "patterns": [],
                },
                {
                    "name": "missing_metadata",
                    "description": "Essential metadata fields are missing",
                    "severity": "low",
                    "score": 0.35,
                    "patterns": [],
                },
                {
                    "name": "inconsistent_creator",
                    "description": "Creator information is inconsistent",
                    "severity": "medium",
                    "score": 0.55,
                    "patterns": [],
                },
            ],
        },
        "structure": {
            "rules": [
                {
                    "name": "inconsistent_headers_footers",
                    "description": "Headers or footers vary across pages",
                    "severity": "medium",
                    "score": 0.5,
                    "patterns": [],
                },
                {
                    "name": "page_numbering_issues",
                    "description": "Page numbers are missing or inconsistent",
                    "severity": "low",
                    "score": 0.4,
                    "patterns": [],
                },
            ],
        },
        "visual": {
            "rules": [
                {
                    "name": "compression_artifacts",
                    "description": "Signs of recompression or image manipulation",
                    "severity": "medium",
                    "score": 0.55,
                    "patterns": [],
                },
                {
                    "name": "resolution_issues",
                    "description": "Unusual resolution patterns suggesting manipulation",
                    "severity": "low",
                    "score": 0.4,
                    "patterns": [],
                },
            ],
        },
    }
}


class MarkerDetector:
    """
    Detects forgery markers across text, numerical, date, structural,
    metadata, and visual dimensions of a document.
    """

    # Date pattern for parsing
    DATE_PARSE_FORMATS: List[str] = [
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%m-%d-%Y",
        "%m.%d.%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
    ]

    # Monetary pattern
    MONETARY_PATTERN: re.Pattern = re.compile(
        r"[$€£¥₹]\s*([\d,]+\.?\d*)"
        r"|([\d,]+\.?\d*)\s*(?:USD|EUR|GBP|JPY|INR)",
        re.IGNORECASE,
    )

    # Number pattern
    NUMBER_PATTERN: re.Pattern = re.compile(r"\b([\d,]+\.?\d*)\b")

    # Date patterns for extraction
    DATE_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
        re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
        re.compile(
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
            r"\.?\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[a-z]*\.?\s+\d{4})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{4})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b((?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
    ]

    # ID patterns
    ID_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b([A-Z]{2,4}[-]?\d{4,12})\b"),
        re.compile(r"\b(\d{6,15})\b"),
        re.compile(
            r"\b((?:INV|INV-|BILL|BILL-|PO|PO-|REF|REF-|NO\.?|NUM\.?|"
            r"ID|ID-|SN|SN-|TXN|TXN-|#)\s*[-]?\d{3,12})\b",
            re.IGNORECASE,
        ),
    ]

    # Suspicious phrases
    SUSPICIOUS_PHRASES: List[str] = [
        "this is a certified copy",
        "certified true copy",
        "original document",
        "notarized copy",
        "authentic document",
        "official document",
        "verified document",
        "for illustrative purposes",
        "sample document",
        "specimen copy",
        "draft copy",
    ]

    def __init__(self, markers_config_path: Optional[str] = None) -> None:
        """
        Initialize the MarkerDetector.

        Args:
            markers_config_path: Path to markers.json configuration file.
        """
        self.rules = self._load_rules(markers_config_path)
        logger.info("MarkerDetector initialized with %d categories",
                     len(self.rules.get("categories", {})))

    def _load_rules(
        self, config_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load marker rules from JSON config file, falling back to defaults.

        Args:
            config_path: Path to the markers.json file.

        Returns:
            Dictionary of marker rules.
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    rules = json.load(f)
                logger.info("Loaded marker rules from %s", config_path)
                return rules
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    "Failed to load marker rules from %s: %s. Using defaults.",
                    config_path,
                    str(e),
                )

        # Try default location
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data",
            "markers.json",
        )
        if os.path.exists(default_path):
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    rules = json.load(f)
                logger.info("Loaded marker rules from %s", default_path)
                return rules
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    "Failed to load marker rules from %s: %s",
                    default_path,
                    str(e),
                )

        logger.info("Using default marker rules")
        return DEFAULT_MARKER_RULES

    def detect_all_markers(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        feature_vector: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect all forgery markers in a document.

        Args:
            text: Extracted document text.
            metadata: Document metadata dictionary.
            stats: Document statistics dictionary.
            feature_vector: Feature vector from FeatureEngineer.

        Returns:
            List of marker dictionaries with detection results.
        """
        if metadata is None:
            metadata = {}
        if stats is None:
            stats = {}
        if feature_vector is None:
            feature_vector = np.array([])

        markers: List[Dict[str, Any]] = []

        # Run all detection categories
        markers.extend(self._detect_textual_markers(text, stats))
        markers.extend(self._detect_numerical_markers(text))
        markers.extend(self._detect_date_markers(text))
        markers.extend(self._detect_id_markers(text, stats))
        markers.extend(self._detect_structural_markers(text, stats, metadata))
        markers.extend(self._detect_metadata_markers(metadata, stats))
        markers.extend(self._detect_visual_markers(stats, feature_vector))

        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        markers.sort(key=lambda m: severity_order.get(m.get("severity", "low"), 3))

        logger.info(
            "Detected %d markers: %d high, %d medium, %d low",
            len(markers),
            sum(1 for m in markers if m["severity"] == "high"),
            sum(1 for m in markers if m["severity"] == "medium"),
            sum(1 for m in markers if m["severity"] == "low"),
        )
        return markers

    def _create_marker(
        self,
        category: str,
        name: str,
        severity: str,
        description: str,
        evidence: str,
        score: float,
        page_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Create a standardized marker dictionary.

        Args:
            category: Marker category.
            name: Marker name.
            severity: Severity level (high/medium/low).
            description: Human-readable description.
            evidence: Evidence string supporting this marker.
            score: Confidence score (0-1).
            page_number: Page number where marker was found.

        Returns:
            Marker dictionary.
        """
        return {
            "category": category,
            "name": name,
            "severity": severity,
            "description": description,
            "evidence": evidence,
            "score": min(max(score, 0.0), 1.0),
            "page_number": page_number,
        }

    # ========== TEXTUAL MARKERS ==========

    def _detect_textual_markers(
        self, text: str, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect textual forgery markers.

        Args:
            text: Document text.
            stats: Document statistics.

        Returns:
            List of textual markers found.
        """
        markers: List[Dict[str, Any]] = []

        # Inconsistent terminology detection
        markers.extend(self._detect_inconsistent_terminology(text))

        # Suspicious phrases
        markers.extend(self._detect_suspicious_phrases(text))

        # Repeated sentences
        markers.extend(self._detect_repeated_sentences(text))

        # Abnormal capitalization
        markers.extend(self._detect_abnormal_capitalization(text, stats))

        # Excessive punctuation
        markers.extend(self._detect_excessive_punctuation(text, stats))

        # Template-like language
        markers.extend(self._detect_template_language(text))

        return markers

    def _detect_inconsistent_terminology(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect inconsistent terminology in the document."""
        markers: List[Dict[str, Any]] = []

        terminology_groups: List[Tuple[str, List[str]]] = [
            ("invoice", ["invoice", "bill", "receipt", "statement"]),
            ("payment", ["payment", "remittance", "settlement", "tender"]),
            ("customer", ["customer", "client", "buyer", "purchaser", "patron"]),
            ("total", ["total", "sum", "amount due", "balance due", "grand total"]),
            ("tax", ["tax", "vat", "gst", "sales tax", "duty"]),
            ("address", ["address", "location", "premises", "residence"]),
            ("date", ["date", "dated", "issue date", "effective date"]),
        ]

        text_lower = text.lower()
        for concept, alternatives in terminology_groups:
            found_terms: List[str] = []
            for term in alternatives:
                if term in text_lower:
                    found_terms.append(term)

            if len(found_terms) > 1:
                evidence = (
                    f"Multiple terms used for same concept: "
                    f"{', '.join(found_terms)}"
                )
                markers.append(
                    self._create_marker(
                        category="textual",
                        name="inconsistent_terminology",
                        severity="medium",
                        description=(
                            "Same concept referred to by different terms "
                            "throughout the document"
                        ),
                        evidence=evidence,
                        score=0.5 + 0.1 * (len(found_terms) - 2),
                    )
                )

        return markers

    def _detect_suspicious_phrases(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect suspicious phrases commonly found in forged documents."""
        markers: List[Dict[str, Any]] = []
        text_lower = text.lower()

        found_phrases: List[str] = []
        for phrase in self.SUSPICIOUS_PHRASES:
            if phrase in text_lower:
                found_phrases.append(phrase)

        if found_phrases:
            evidence = f"Suspicious phrases found: {'; '.join(found_phrases)}"
            score = min(0.5 + 0.1 * len(found_phrases), 0.9)
            markers.append(
                self._create_marker(
                    category="textual",
                    name="suspicious_phrases",
                    severity="medium" if len(found_phrases) < 3 else "high",
                    description=(
                        "Document contains phrases commonly found in "
                        "forged or template documents"
                    ),
                    evidence=evidence,
                    score=score,
                )
            )

        return markers

    def _detect_repeated_sentences(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect sentences that are repeated verbatim."""
        markers: List[Dict[str, Any]] = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        sentence_counts: Dict[str, int] = {}
        for sentence in sentences:
            normalized = " ".join(sentence.lower().split())
            sentence_counts[normalized] = sentence_counts.get(normalized, 0) + 1

        repeated = {
            s: count
            for s, count in sentence_counts.items()
            if count > 1
        }

        if repeated:
            max_repeat = max(repeated.values())
            evidence_parts = [
                f'"{s[:80]}..." (repeated {c}x)'
                for s, c in sorted(
                    repeated.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]
            evidence = "Repeated sentences: " + "; ".join(evidence_parts)
            score = min(0.4 + 0.05 * len(repeated) + 0.1 * (max_repeat - 2), 0.85)

            markers.append(
                self._create_marker(
                    category="textual",
                    name="repeated_sentences",
                    severity="medium" if max_repeat <= 3 else "high",
                    description=(
                        f"Found {len(repeated)} sentences repeated "
                        f"(max {max_repeat} times)"
                    ),
                    evidence=evidence,
                    score=score,
                )
            )

        return markers

    def _detect_abnormal_capitalization(
        self, text: str, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect unusual capitalization patterns."""
        markers: List[Dict[str, Any]] = []

        lines = text.split("\n")
        all_caps_lines = 0
        total_content_lines = 0

        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                total_content_lines += 1
                if stripped.isupper():
                    all_caps_lines += 1

        if total_content_lines > 0:
            caps_ratio = all_caps_lines / total_content_lines
            if caps_ratio > 0.4:
                severity = "high" if caps_ratio > 0.6 else "medium"
                score = min(0.3 + caps_ratio, 0.8)
                markers.append(
                    self._create_marker(
                        category="textual",
                        name="abnormal_capitalization",
                        severity=severity,
                        description=(
                            f"Abnormal capitalization: "
                            f"{all_caps_lines}/{total_content_lines} lines "
                            f"are all-caps"
                        ),
                        evidence=(
                            f"All-caps ratio: {caps_ratio:.1%} "
                            f"({all_caps_lines} of {total_content_lines} lines)"
                        ),
                        score=score,
                    )
                )

        return markers

    def _detect_excessive_punctuation(
        self, text: str, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect excessive or unusual punctuation usage."""
        markers: List[Dict[str, Any]] = []

        # Count punctuation
        exclamation_count = text.count("!")
        question_count = text.count("?")
        ellipsis_count = len(re.findall(r"\.\.\.", text))

        if exclamation_count > 10:
            markers.append(
                self._create_marker(
                    category="textual",
                    name="excessive_punctuation",
                    severity="low",
                    description="Excessive use of exclamation marks",
                    evidence=(
                        f"Found {exclamation_count} exclamation marks "
                        f"in document"
                    ),
                    score=min(0.3 + exclamation_count * 0.02, 0.7),
                )
            )

        if ellipsis_count > 5:
            markers.append(
                self._create_marker(
                    category="textual",
                    name="excessive_punctuation",
                    severity="low",
                    description="Excessive use of ellipsis",
                    evidence=f"Found {ellipsis_count} ellipsis sequences",
                    score=min(0.3 + ellipsis_count * 0.03, 0.7),
                )
            )

        return markers

    def _detect_template_language(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect template-like language patterns."""
        markers: List[Dict[str, Any]] = []

        template_patterns = [
            (r"\[.*?(?:name|date|address|amount).*?\]", "Bracket placeholders"),
            (r"\bfill\s+in\s+(?:the\s+)?(?:below|following)\b", "Fill-in instructions"),
            (r"\binsert\s+(?:your|the)\b", "Insert instructions"),
            (r"\b(?:enter|provide)\s+(?:your|the)\b", "Enter/provide instructions"),
            (r"_{3,}", "Underline blanks (3+ underscores)"),
            (r"\bXX+\b", "Placeholder X marks"),
            (r"\bN/A\b.*\bN/A\b", "Multiple N/A entries"),
        ]

        found_templates: List[str] = []
        for pattern, desc in template_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_templates.append(
                    f"{desc} ({len(matches)} occurrences)"
                )

        if found_templates:
            evidence = "Template indicators: " + "; ".join(found_templates)
            markers.append(
                self._create_marker(
                    category="textual",
                    name="template_like_language",
                    severity="medium",
                    description=(
                        "Document contains template-like language patterns"
                    ),
                    evidence=evidence,
                    score=min(0.4 + 0.05 * len(found_templates), 0.8),
                )
            )

        return markers

    # ========== NUMERICAL MARKERS ==========

    def _detect_numerical_markers(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect numerical forgery markers."""
        markers: List[Dict[str, Any]] = []

        markers.extend(self._detect_inconsistent_totals(text))
        markers.extend(self._detect_amount_mismatches(text))
        markers.extend(self._detect_percentage_inconsistencies(text))
        markers.extend(self._detect_repeated_numbers(text))

        return markers

    def _parse_monetary_value(self, value_str: str) -> Optional[float]:
        """Parse a monetary value string to float."""
        cleaned = re.sub(r"[,$€£¥₹\s]", "", value_str)
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _detect_inconsistent_totals(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect arithmetic inconsistencies in totals."""
        markers: List[Dict[str, Any]] = []

        # Look for subtotal, tax, and total patterns
        subtotal_pattern = re.compile(
            r"(?:sub\s*total|subtotal)[:\s]*[$€£¥₹]?\s*([\d,]+\.?\d*)",
            re.IGNORECASE,
        )
        tax_pattern = re.compile(
            r"(?:tax|vat|gst)[:\s]*[$€£¥₹]?\s*([\d,]+\.?\d*)",
            re.IGNORECASE,
        )
        total_pattern = re.compile(
            r"(?:total|grand\s*total|amount\s*due|balance\s*due)[:\s]*"
            r"[$€£¥₹]?\s*([\d,]+\.?\d*)",
            re.IGNORECASE,
        )

        subtotals = subtotal_pattern.findall(text)
        taxes = tax_pattern.findall(text)
        totals = total_pattern.findall(text)

        if subtotals and taxes and totals:
            for sub_str in subtotals:
                sub_val = self._parse_monetary_value(sub_str)
                if sub_val is None:
                    continue
                for tax_str in taxes:
                    tax_val = self._parse_monetary_value(tax_str)
                    if tax_val is None:
                        continue
                    expected_total = sub_val + tax_val
                    for total_str in totals:
                        total_val = self._parse_monetary_value(total_str)
                        if total_val is None:
                            continue
                        if expected_total > 0:
                            diff_pct = abs(expected_total - total_val) / expected_total
                            if diff_pct > 0.01:  # More than 1% off
                                evidence = (
                                    f"Subtotal ({sub_val}) + Tax ({tax_val}) = "
                                    f"{expected_total}, but Total shows "
                                    f"{total_val} "
                                    f"(difference: {diff_pct:.1%})"
                                )
                                severity = (
                                    "high" if diff_pct > 0.05 else "medium"
                                )
                                markers.append(
                                    self._create_marker(
                                        category="numerical",
                                        name="inconsistent_totals",
                                        severity=severity,
                                        description=(
                                            "Subtotal + tax does not equal "
                                            "stated total"
                                        ),
                                        evidence=evidence,
                                        score=min(0.6 + diff_pct * 2, 0.95),
                                    )
                                )

        return markers

    def _detect_amount_mismatches(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect line items with inconsistent amounts."""
        markers: List[Dict[str, Any]] = []

        # Look for patterns like "Item ... $X.XX ... $Y.YY" on same line
        line_pattern = re.compile(
            r"(.+?)\s+[$€£¥₹]?\s*([\d,]+\.?\d*)\s+.*?[$€£¥₹]?\s*([\d,]+\.?\d*)",
            re.IGNORECASE,
        )

        mismatches: List[str] = []
        for match in line_pattern.finditer(text):
            item_name = match.group(1).strip()
            val1 = self._parse_monetary_value(match.group(2))
            val2 = self._parse_monetary_value(match.group(3))
            if val1 is not None and val2 is not None and val1 != val2:
                mismatches.append(
                    f'"{item_name[:50]}": {val1} vs {val2}'
                )

        if mismatches:
            evidence = (
                "Amount mismatches found: " + "; ".join(mismatches[:5])
            )
            markers.append(
                self._create_marker(
                    category="numerical",
                    name="amount_mismatches",
                    severity="high",
                    description=(
                        f"Found {len(mismatches)} line items with "
                        f"different amounts"
                    ),
                    evidence=evidence,
                    score=min(0.6 + 0.05 * len(mismatches), 0.9),
                )
            )

        return markers

    def _detect_percentage_inconsistencies(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect percentage values that don't add up correctly."""
        markers: List[Dict[str, Any]] = []

        # Find percentage values
        pct_pattern = re.compile(r"([\d.]+)\s*%")
        pct_matches = pct_pattern.findall(text)

        if pct_matches:
            # Check for allocation percentages that should sum to 100
            allocation_contexts = [
                "allocation",
                "distribution",
                "breakdown",
                "split",
                "share",
                "proportion",
            ]

            text_lower = text.lower()
            for context in allocation_contexts:
                if context in text_lower:
                    # Find percentages near this context
                    idx = text_lower.index(context)
                    window = text[max(0, idx - 200) : idx + 500]
                    window_pcts = [
                        float(p) for p in pct_pattern.findall(window)
                    ]

                    if len(window_pcts) > 1:
                        total_pct = sum(window_pcts)
                        if (
                            abs(total_pct - 100) > 0.1
                            and total_pct > 0
                        ):
                            evidence = (
                                f"Percentages near '{context}' section sum to "
                                f"{total_pct:.1f}% instead of 100% "
                                f"(values: {window_pcts})"
                            )
                            markers.append(
                                self._create_marker(
                                    category="numerical",
                                    name="percentage_inconsistencies",
                                    severity="medium",
                                    description=(
                                        "Allocation percentages do not sum to 100%"
                                    ),
                                    evidence=evidence,
                                    score=min(
                                        0.5
                                        + abs(total_pct - 100) * 0.02,
                                        0.85,
                                    ),
                                )
                            )

        return markers

    def _detect_repeated_numbers(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect suspiciously repeated numerical values."""
        markers: List[Dict[str, Any]] = []

        # Extract all standalone numbers (at least 4 digits)
        big_numbers = re.findall(r"\b(\d{4,})\b", text)
        if not big_numbers:
            return markers

        number_counts: Dict[str, int] = {}
        for num in big_numbers:
            number_counts[num] = number_counts.get(num, 0) + 1

        repeated = {
            num: count
            for num, count in number_counts.items()
            if count > 2
        }

        if repeated:
            evidence_parts = [
                f"{num} (appears {count}x)"
                for num, count in sorted(
                    repeated.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]
            evidence = "Repeated numbers: " + "; ".join(evidence_parts)
            max_repeat = max(repeated.values())

            markers.append(
                self._create_marker(
                    category="numerical",
                    name="repeated_numbers",
                    severity="medium",
                    description=(
                        f"Found {len(repeated)} numbers repeated 3+ times"
                    ),
                    evidence=evidence,
                    score=min(0.4 + max_repeat * 0.05, 0.8),
                )
            )

        return markers

    # ========== DATE MARKERS ==========

    def _detect_date_markers(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect date-related forgery markers."""
        markers: List[Dict[str, Any]] = []

        markers.extend(self._detect_impossible_dates(text))
        markers.extend(self._detect_future_dates(text))
        markers.extend(self._detect_date_order_issues(text))
        markers.extend(self._detect_inconsistent_date_formats(text))

        return markers

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Attempt to parse a date string."""
        for fmt in self.DATE_PARSE_FORMATS:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _detect_impossible_dates(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect dates that cannot exist."""
        markers: List[Dict[str, Any]] = []

        impossible_patterns = [
            (
                re.compile(
                    r"\b(02[/-](?:30|31)[/-]\d{2,4})\b", re.IGNORECASE
                ),
                "February 30th or 31st (impossible date)",
            ),
            (
                re.compile(
                    r"\b(0[469]|11)[/-]31[/-]\d{2,4}\b"
                ),
                "31st day of a 30-day month",
            ),
            (
                re.compile(
                    r"\b(02[/-]29[/-](?:[1-9]\d(?!00$)|(?<!\d00)\d{2}(?!00$)))\b"
                ),
                "February 29 in a non-leap year",
            ),
            (
                re.compile(
                    r"\b(1[3-9][/-]\d{1,2}[/-]\d{2,4})\b"
                ),
                "Month greater than 12",
            ),
            (
                re.compile(
                    r"\b(\d{1,2}[/-]1[3-9][/-]\d{2,4})\b"
                ),
                "Day greater than 31",
            ),
        ]

        for pattern, desc in impossible_patterns:
            matches = pattern.findall(text)
            if matches:
                evidence = f"{desc}: {', '.join(matches[:3])}"
                markers.append(
                    self._create_marker(
                        category="date",
                        name="impossible_dates",
                        severity="high",
                        description="Document contains impossible dates",
                        evidence=evidence,
                        score=0.9,
                    )
                )

        return markers

    def _detect_future_dates(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect dates that are significantly in the future."""
        markers: List[Dict[str, Any]] = []

        now = datetime.now()
        future_threshold = now.year + 5  # More than 5 years in future

        for pattern in self.DATE_PATTERNS:
            matches = pattern.findall(text)
            for date_str in matches:
                parsed = self._parse_date(date_str)
                if parsed and parsed.year > future_threshold:
                    evidence = (
                        f"Future date found: {date_str} "
                        f"({parsed.year}, "
                        f"{parsed.year - now.year} years from now)"
                    )
                    markers.append(
                        self._create_marker(
                            category="date",
                            name="future_dates",
                            severity="medium",
                            description="Document contains dates far in the future",
                            evidence=evidence,
                            score=min(
                                0.5 + (parsed.year - future_threshold) * 0.02,
                                0.8,
                            ),
                        )
                    )

        return markers

    def _detect_date_order_issues(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect issue date after expiry date."""
        markers: List[Dict[str, Any]] = []

        # Look for issue/expiry pairs
        issue_pattern = re.compile(
            r"(?:issue|issue\s*date|effective|from|start)[:\s]*"
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            re.IGNORECASE,
        )
        expiry_pattern = re.compile(
            r"(?:expir|expir\w*|valid\s*until|until|end|expiry)[:\s]*"
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            re.IGNORECASE,
        )

        issues = issue_pattern.findall(text)
        expiries = expiry_pattern.findall(text)

        for issue_str in issues:
            issue_date = self._parse_date(issue_str)
            if issue_date is None:
                continue
            for expiry_str in expiries:
                expiry_date = self._parse_date(expiry_str)
                if expiry_date is None:
                    continue
                if issue_date > expiry_date:
                    evidence = (
                        f"Issue date ({issue_str}) is after "
                        f"expiry date ({expiry_str})"
                    )
                    markers.append(
                        self._create_marker(
                            category="date",
                            name="issue_after_expiry",
                            severity="high",
                            description=(
                                "Issue date appears after expiry date"
                            ),
                            evidence=evidence,
                            score=0.85,
                        )
                    )

        return markers

    def _detect_inconsistent_date_formats(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect inconsistent date formatting in document."""
        markers: List[Dict[str, Any]] = []

        format_groups: Dict[str, List[str]] = {}

        format_defs = [
            ("MM/DD/YYYY", re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")),
            ("DD/MM/YYYY", re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")),
            ("YYYY-MM-DD", re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")),
            ("DD-MM-YYYY", re.compile(r"\b\d{1,2}-\d{1,2}-\d{4}\b")),
            ("Month DD, YYYY", re.compile(
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
                re.IGNORECASE,
            )),
            ("DD Month YYYY", re.compile(
                r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]*\.?\s+\d{4}\b",
                re.IGNORECASE,
            )),
        ]

        for fmt_name, pattern in format_defs:
            matches = pattern.findall(text)
            if matches:
                format_groups[fmt_name] = matches

        if len(format_groups) > 1:
            fmt_list = list(format_groups.keys())
            evidence = (
                f"Multiple date formats used: {', '.join(fmt_list)}"
            )
            markers.append(
                self._create_marker(
                    category="date",
                    name="inconsistent_date_formats",
                    severity="low",
                    description=(
                        "Document uses inconsistent date formatting"
                    ),
                    evidence=evidence,
                    score=0.3 + 0.05 * (len(format_groups) - 1),
                )
            )

        return markers

    # ========== ID/REFERENCE MARKERS ==========

    def _detect_id_markers(
        self, text: str, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect ID and reference number markers."""
        markers: List[Dict[str, Any]] = []

        markers.extend(self._detect_repeated_ids(text))
        markers.extend(self._detect_malformed_ids(text))
        markers.extend(self._detect_id_inconsistencies(text))

        return markers

    def _detect_repeated_ids(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect ID numbers that are repeated (potentially copy-paste)."""
        markers: List[Dict[str, Any]] = []

        all_ids: List[str] = []
        for pattern in self.ID_PATTERNS:
            all_ids.extend(pattern.findall(text))

        if not all_ids:
            return markers

        id_counts: Dict[str, int] = {}
        for id_val in all_ids:
            normalized = id_val.strip().upper()
            id_counts[normalized] = id_counts.get(normalized, 0) + 1

        repeated = {
            id_val: count
            for id_val, count in id_counts.items()
            if count > 1
        }

        if repeated:
            evidence_parts = [
                f"{id_val} (appears {count}x)"
                for id_val, count in sorted(
                    repeated.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]
            evidence = "Repeated IDs: " + "; ".join(evidence_parts)
            markers.append(
                self._create_marker(
                    category="id_reference",
                    name="repeated_ids",
                    severity="medium",
                    description=(
                        f"Found {len(repeated)} ID/reference numbers "
                        f"appearing multiple times"
                    ),
                    evidence=evidence,
                    score=min(0.5 + 0.1 * len(repeated), 0.8),
                )
            )

        return markers

    def _detect_malformed_ids(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect ID numbers with malformed patterns."""
        markers: List[Dict[str, Any]] = []

        # Look for IDs that break expected patterns
        # e.g., all zeros, sequential digits, etc.
        suspicious_ids: List[Tuple[str, str]] = []

        for pattern in self.ID_PATTERNS:
            for match in pattern.finditer(text):
                id_val = match.group(1) if match.lastindex else match.group()
                id_clean = re.sub(r"[^a-zA-Z0-9]", "", id_val)

                # All same digit
                if len(id_clean) > 3 and len(set(id_clean)) == 1:
                    suspicious_ids.append(
                        (id_val, f"All characters are '{id_clean[0]}'")
                    )
                # Sequential digits
                elif len(id_clean) > 4 and id_clean.isdigit():
                    is_sequential = all(
                        int(id_clean[i + 1]) - int(id_clean[i]) == 1
                        for i in range(len(id_clean) - 1)
                    )
                    if is_sequential:
                        suspicious_ids.append(
                            (id_val, "Sequential digits")
                        )

        if suspicious_ids:
            evidence_parts = [
                f"{id_val} ({reason})"
                for id_val, reason in suspicious_ids[:5]
            ]
            evidence = "Malformed IDs: " + "; ".join(evidence_parts)
            markers.append(
                self._create_marker(
                    category="id_reference",
                    name="malformed_patterns",
                    severity="medium",
                    description="ID numbers with suspicious patterns",
                    evidence=evidence,
                    score=0.7,
                )
            )

        return markers

    def _detect_id_inconsistencies(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect inconsistent ID formats for same document type."""
        markers: List[Dict[str, Any]] = []

        # Group IDs by prefix
        prefix_groups: Dict[str, List[str]] = {}

        for pattern in self.ID_PATTERNS:
            for match in pattern.finditer(text):
                id_val = match.group(1) if match.lastindex else match.group()
                prefix_match = re.match(r"^([A-Za-z-]+)", id_val)
                if prefix_match:
                    prefix = prefix_match.group(1).upper().rstrip("-")
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(id_val)

        # Check for inconsistent formats within same prefix
        for prefix, ids in prefix_groups.items():
            if len(ids) > 1:
                lengths = [len(re.sub(r"[^a-zA-Z0-9]", "", i)) for i in ids]
                if len(set(lengths)) > 1:
                    evidence = (
                        f"IDs with prefix '{prefix}' have inconsistent "
                        f"formats: {', '.join(ids[:5])}"
                    )
                    markers.append(
                        self._create_marker(
                            category="id_reference",
                            name="inconsistent_ids",
                            severity="medium",
                            description=(
                                f"ID numbers with prefix '{prefix}' have "
                                f"inconsistent formats"
                            ),
                            evidence=evidence,
                            score=0.65,
                        )
                    )

        return markers

    # ========== STRUCTURAL MARKERS ==========

    def _detect_structural_markers(
        self,
        text: str,
        stats: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect structural forgery markers."""
        markers: List[Dict[str, Any]] = []

        markers.extend(self._detect_missing_sections(text))
        markers.extend(self._detect_inconsistent_headers_footers(text, stats))
        markers.extend(self._detect_page_numbering_issues(text, stats))

        return markers

    def _detect_missing_sections(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Detect potentially missing sections in the document."""
        markers: List[Dict[str, Any]] = []

        # Common section expectations for different document types
        document_sections = {
            "invoice": [
                (r"invoice\s*(?:number|#|no)", "Invoice number"),
                (r"date\s*(?:of\s*issue|issued)", "Issue date"),
                (r"(?:bill\s*to|ship\s*to|sold\s*to)", "Recipient information"),
                (r"(?:item|description|product)", "Line items"),
                (r"(?:total|amount\s*due)", "Total amount"),
                (r"(?:payment\s*terms|terms)", "Payment terms"),
            ],
            "contract": [
                (r"(?:parties|between)", "Parties"),
                (r"(?:terms|conditions)", "Terms"),
                (r"(?:signature|signed)", "Signature"),
                (r"(?:date|effective)", "Effective date"),
                (r"(?:obligations|responsibilities)", "Obligations"),
            ],
            "certificate": [
                (r"(?:certif|this\s*is\s*to\s*certify)", "Certification statement"),
                (r"(?:name|recipient)", "Recipient"),
                (r"(?:date|issued)", "Date"),
                (r"(?:signature|authorized)", "Authorization"),
                (r"(?:seal|stamp)", "Seal/stamp"),
            ],
        }

        text_lower = text.lower()

        for doc_type, sections in document_sections.items():
            # Only check if document appears to be this type
            if doc_type in text_lower or (
                doc_type == "invoice"
                and any(
                    w in text_lower
                    for w in ["invoice", "bill", "receipt", "amount"]
                )
            ):
                missing: List[str] = []
                for pattern, section_name in sections:
                    if not re.search(pattern, text_lower):
                        missing.append(section_name)

                if missing and len(missing) >= 2:
                    evidence = (
                        f"Potentially missing sections: "
                        f"{', '.join(missing)}"
                    )
                    markers.append(
                        self._create_marker(
                            category="structural",
                            name="missing_sections",
                            severity="medium",
                            description=(
                                f"Document may be missing "
                                f"{len(missing)} expected sections"
                            ),
                            evidence=evidence,
                            score=min(
                                0.4 + len(missing) * 0.05, 0.75
                            ),
                        )
                    )

        return markers

    def _detect_inconsistent_headers_footers(
        self, text: str, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect inconsistent headers or footers."""
        markers: List[Dict[str, Any]] = []

        lines = text.split("\n")
        if len(lines) < 10:
            return markers

        # Check first and last lines of each "page" (approximate by line groups)
        header_lines = [
            lines[i].strip()
            for i in range(min(3, len(lines)))
            if lines[i].strip()
        ]
        footer_lines = [
            lines[i].strip()
            for i in range(max(0, len(lines) - 3), len(lines))
            if lines[i].strip()
        ]

        # Look for page break indicators
        page_breaks = [
            i
            for i, line in enumerate(lines)
            if re.match(
                r"^(?:page\s*\d+|---|\*\*\*|===|\f)$",
                line.strip(),
                re.IGNORECASE,
            )
        ]

        if len(page_breaks) > 1:
            # Check for inconsistent content around page breaks
            around_breaks = []
            for bp in page_breaks:
                context = lines[max(0, bp - 2) : bp + 3]
                around_breaks.append(" ".join(c.strip() for c in context))

            if len(set(around_breaks)) > 1:
                markers.append(
                    self._create_marker(
                        category="structural",
                        name="inconsistent_headers_footers",
                        severity="medium",
                        description=(
                            "Headers or footers appear inconsistent "
                            "across pages"
                        ),
                        evidence=(
                            f"Found {len(page_breaks)} page breaks "
                            f"with varying surrounding content"
                        ),
                        score=0.5,
                    )
                )

        return markers

    def _detect_page_numbering_issues(
        self, text: str, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect page numbering issues."""
        markers: List[Dict[str, Any]] = []

        page_count = max(int(stats.get("page_count", 1)), 1)

        # Look for page number references
        page_num_pattern = re.compile(
            r"(?:page|p\.?)\s*(\d+)", re.IGNORECASE
        )
        page_refs = page_num_pattern.findall(text)

        if page_count > 1 and not page_refs:
            markers.append(
                self._create_marker(
                    category="structural",
                    name="page_numbering_issues",
                    severity="low",
                    description="Multi-page document has no page numbers",
                    evidence=(
                        f"Document has {page_count} pages but no "
                        f"page number references found"
                    ),
                    score=0.4,
                )
            )
        elif page_refs:
            max_ref = max(int(p) for p in page_refs)
            if max_ref != page_count and abs(max_ref - page_count) > 1:
                markers.append(
                    self._create_marker(
                        category="structural",
                        name="page_numbering_issues",
                        severity="medium",
                        description="Page numbers don't match page count",
                        evidence=(
                            f"Max page reference is {max_ref} but "
                            f"document has {page_count} pages"
                        ),
                        score=0.55,
                    )
                )

        return markers

    # ========== METADATA MARKERS ==========

    def _detect_metadata_markers(
        self,
        metadata: Dict[str, Any],
        stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect metadata-related forgery markers."""
        markers: List[Dict[str, Any]] = []

        markers.extend(self._detect_suspicious_timestamps(metadata))
        markers.extend(self._detect_missing_metadata(metadata))
        markers.extend(self._detect_inconsistent_creator(metadata))

        return markers

    def _detect_suspicious_timestamps(
        self, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect suspicious creation/modification timestamps."""
        markers: List[Dict[str, Any]] = []

        creation = metadata.get("creation_date")
        modification = metadata.get("modification_date")

        if creation and modification:
            # Check if they are suspiciously close (within 1 second)
            # This could indicate automated document generation
            try:
                if isinstance(creation, str):
                    creation_dt = self._parse_date(creation)
                elif isinstance(creation, datetime):
                    creation_dt = creation
                else:
                    creation_dt = None

                if isinstance(modification, str):
                    modification_dt = self._parse_date(modification)
                elif isinstance(modification, datetime):
                    modification_dt = modification
                else:
                    modification_dt = None

                if creation_dt and modification_dt:
                    diff = abs(
                        (modification_dt - creation_dt).total_seconds()
                    )
                    if diff < 1.0:
                        markers.append(
                            self._create_marker(
                                category="metadata",
                                name="suspicious_timestamps",
                                severity="medium",
                                description=(
                                    "Creation and modification times "
                                    "are suspiciously close"
                                ),
                                evidence=(
                                    f"Time difference between creation "
                                    f"and modification: {diff:.2f} seconds"
                                ),
                                score=0.5,
                            )
                        )
                    elif diff < 5.0:
                        markers.append(
                            self._create_marker(
                                category="metadata",
                                name="suspicious_timestamps",
                                severity="low",
                                description=(
                                    "Creation and modification times "
                                    "are very close"
                                ),
                                evidence=(
                                    f"Time difference between creation "
                                    f"and modification: {diff:.1f} seconds"
                                ),
                                score=0.35,
                            )
                        )
            except (ValueError, TypeError) as e:
                logger.debug(
                    "Error parsing timestamps: %s", str(e)
                )

        return markers

    def _detect_missing_metadata(
        self, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect missing essential metadata fields."""
        markers: List[Dict[str, Any]] = []

        essential_fields = {
            "title": "Document title",
            "author": "Author",
            "creator": "Creator application",
            "creation_date": "Creation date",
        }

        missing: List[str] = []
        for field, description in essential_fields.items():
            if not metadata.get(field):
                missing.append(description)

        if missing:
            markers.append(
                self._create_marker(
                    category="metadata",
                    name="missing_metadata",
                    severity="low",
                    description=(
                        f"Document is missing {len(missing)} "
                        f"essential metadata fields"
                    ),
                    evidence=(
                        f"Missing fields: {', '.join(missing)}"
                    ),
                    score=0.2 + 0.05 * len(missing),
                )
            )

        return markers

    def _detect_inconsistent_creator(
        self, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect inconsistent creator information."""
        markers: List[Dict[str, Any]] = []

        creator = metadata.get("creator", "")
        producer = metadata.get("producer", "")
        author = metadata.get("author", "")

        # Check if creator and producer suggest different applications
        if creator and producer:
            creator_lower = creator.lower()
            producer_lower = producer.lower()

            # Extract application names
            creator_words = set(creator_lower.split())
            producer_words = set(producer_lower.split())

            # If they share no common words, it might be suspicious
            if (
                creator_words
                and producer_words
                and not creator_words & producer_words
                and len(creator_lower) > 5
                and len(producer_lower) > 5
            ):
                markers.append(
                    self._create_marker(
                        category="metadata",
                        name="inconsistent_creator",
                        severity="medium",
                        description=(
                            "Creator and producer metadata suggest "
                            "different applications"
                        ),
                        evidence=(
                            f"Creator: '{creator}', "
                            f"Producer: '{producer}'"
                        ),
                        score=0.55,
                    )
                )

        return markers

    # ========== VISUAL MARKERS ==========

    def _detect_visual_markers(
        self,
        stats: Dict[str, Any],
        feature_vector: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Detect visual forgery markers."""
        markers: List[Dict[str, Any]] = []

        markers.extend(self._detect_compression_artifacts(stats))
        markers.extend(self._detect_resolution_issues(stats))

        return markers

    def _detect_compression_artifacts(
        self, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect signs of recompression or image manipulation."""
        markers: List[Dict[str, Any]] = []

        suspicious_regions = stats.get("suspicious_region_count", 0)
        if suspicious_regions > 0:
            markers.append(
                self._create_marker(
                    category="visual",
                    name="compression_artifacts",
                    severity="medium" if suspicious_regions < 3 else "high",
                    description=(
                        f"Found {suspicious_regions} regions with "
                        f"suspicious compression patterns"
                    ),
                    evidence=(
                        f"{suspicious_regions} suspicious regions detected "
                        f"possibly indicating image manipulation"
                    ),
                    score=min(0.4 + suspicious_regions * 0.1, 0.85),
                )
            )

        return markers

    def _detect_resolution_issues(
        self, stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect unusual resolution patterns."""
        markers: List[Dict[str, Any]] = []

        resolution = stats.get("resolution")
        if resolution and isinstance(resolution, (list, tuple)):
            if len(resolution) == 2:
                width, height = resolution
                # Very low resolution
                if width < 300 or height < 300:
                    markers.append(
                        self._create_marker(
                            category="visual",
                            name="resolution_issues",
                            severity="low",
                            description="Document has unusually low resolution",
                            evidence=(
                                f"Resolution: {width}x{height} "
                                f"(below 300px threshold)"
                            ),
                            score=0.4,
                        )
                    )

        ocr_confidence = stats.get("ocr_confidence", 1.0)
        if ocr_confidence < 0.6:
            markers.append(
                self._create_marker(
                    category="visual",
                    name="resolution_issues",
                    severity="medium",
                    description=(
                        "Low OCR confidence suggests quality issues"
                    ),
                    evidence=(
                        f"OCR confidence: {ocr_confidence:.1%} "
                        f"(below 60% threshold)"
                    ),
                    score=min(0.5 + (0.6 - ocr_confidence), 0.8),
                )
            )

        return markers
