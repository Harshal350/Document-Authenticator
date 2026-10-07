"""
Feature extraction system for DocuGuard fake document detection.
Extracts comprehensive text, structural, numerical, OCR, metadata, and visual features.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Extracts features from document text, metadata, and statistical information
    for fake document detection.
    """

    # Common suspicious phrases that may indicate template/forged documents
    SUSPICIOUS_PHRASES: List[str] = [
        "this is a certified copy",
        "certified true copy",
        "original document",
        "notarized copy",
        "authentic document",
        "official document",
        "verified document",
        "do not duplicate",
        "void if altered",
        "security feature",
        "this document is",
        "any resemblance",
        "for illustrative purposes",
        "sample document",
        "specimen copy",
        "draft copy",
        "this is not a legal document",
        "terms and conditions apply",
        "subject to verification",
        "pending verification",
        "not valid without seal",
        "invalid without stamp",
        "photocopy not accepted",
        "must be original",
        "color copy required",
        "do not fold",
        "handle with care",
    ]

    # Feature names in extraction order for the feature vector
    FEATURE_NAMES: List[str] = [
        # Text features
        "word_count",
        "char_count",
        "sentence_count",
        "avg_sentence_length",
        "vocabulary_richness",
        "punctuation_ratio",
        "uppercase_ratio",
        "digit_ratio",
        "special_char_ratio",
        "repeated_phrase_count",
        "spelling_anomaly_count",
        "suspicious_phrase_count",
        # Structural features
        "page_count",
        "paragraph_count",
        "table_count",
        "heading_count",
        "blank_page_count",
        "font_size_variance",
        "alignment_variance",
        # Numerical features
        "number_count",
        "monetary_values_count",
        "date_count",
        "total_inconsistencies",
        "id_inconsistencies",
        # OCR features
        "ocr_confidence",
        "low_confidence_text_pct",
        # Metadata features
        "has_author",
        "has_creator",
        "has_dates",
        "metadata_completeness",
        # Visual features
        "image_count",
        "suspicious_region_count",
    ]

    def __init__(self) -> None:
        """Initialize the FeatureEngineer with compiled regex patterns."""
        self._compile_patterns()
        logger.info("FeatureEngineer initialized")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for feature extraction."""
        # Date patterns (various formats)
        self.date_patterns: List[re.Pattern] = [
            re.compile(
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            ),  # MM/DD/YYYY, DD-MM-YY
            re.compile(
                r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
            ),  # YYYY-MM-DD
            re.compile(
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
                r"\.?\s+\d{1,2},?\s+\d{4}\b",
                re.IGNORECASE,
            ),  # Month DD, YYYY
            re.compile(
                r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]*\.?\s+\d{4}\b",
                re.IGNORECASE,
            ),  # DD Month YYYY
            re.compile(
                r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
                r"August|September|October|November|December)\s+\d{4}\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                re.IGNORECASE,
            ),
        ]

        # Monetary value patterns
        self.monetary_pattern: re.Pattern = re.compile(
            r"[$€£¥₹]\s*\d[\d,]*\.?\d*"
            r"|\b\d[\d,]*\.?\d*\s*(?:USD|EUR|GBP|JPY|INR)\b"
            r"|\b(?:dollars?|euros?|pounds?|yen|rupees?)\s*\d[\d,]*\.?\d*",
            re.IGNORECASE,
        )

        # Number patterns (standalone numbers, possibly with commas)
        self.number_pattern: re.Pattern = re.compile(
            r"\b\d[\d,]*\.?\d*\b"
        )

        # ID-like patterns (long digit sequences, alphanumeric codes)
        self.id_patterns: List[re.Pattern] = [
            re.compile(r"\b[A-Z]{2,4}[-]?\d{4,12}\b"),  # ABC1234567 or AB-123456
            re.compile(r"\b\d{6,15}\b"),  # Long digit sequences
            re.compile(
                r"\b[A-Z0-9]{8,20}\b"
            ),  # Alphanumeric codes
            re.compile(
                r"\b(?:INV|INV-|BILL|BILL-|PO|PO-|REF|REF-|NO\.?|NUM\.?|"
                r"ID|ID-|SN|SN-|TXN|TXN-|#)\s*[-]?\d{3,12}\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b\d{3}[-]?\d{3}[-]?\d{3}\b"
            ),  # XXX-XXX-XXX format
        ]

        # Suspicious phrase pattern
        self.suspicious_phrase_pattern: re.Pattern = re.compile(
            r"(?i)\b(?:" + "|".join(
                re.escape(p) for p in self.SUSPICIOUS_PHRASES
            ) + r")\b"
        )

        # Sentence boundaries
        self.sentence_pattern: re.Pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])"
        )

        # Word pattern
        self.word_pattern: re.Pattern = re.compile(r"\b[a-zA-Z]+\b")

    def extract_features(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all features from document text, metadata, and statistics.

        Args:
            text: Extracted document text content.
            metadata: Document metadata dictionary.
            stats: Document statistics dictionary.

        Returns:
            Dictionary of feature name to feature value.
        """
        if metadata is None:
            metadata = {}
        if stats is None:
            stats = {}

        features: Dict[str, Any] = {}

        # Extract each feature category
        text_features = self._extract_text_features(text)
        features.update(text_features)

        structural_features = self._extract_structural_features(text, stats)
        features.update(structural_features)

        numerical_features = self._extract_numerical_features(text, stats)
        features.update(numerical_features)

        ocr_features = self._extract_ocr_features(stats)
        features.update(ocr_features)

        metadata_features = self._extract_metadata_features(metadata)
        features.update(metadata_features)

        visual_features = self._extract_visual_features(stats)
        features.update(visual_features)

        logger.info(
            "Extracted %d features from document", len(features)
        )
        return features

    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """
        Extract text-based features from document content.

        Args:
            text: Document text content.

        Returns:
            Dictionary of text features.
        """
        features: Dict[str, float] = {}

        # Basic counts
        features["word_count"] = float(len(text.split()))
        features["char_count"] = float(len(text))

        # Sentence analysis
        sentences = self.sentence_pattern.split(text.strip())
        if not sentences or (len(sentences) == 1 and not sentences[0].strip()):
            sentences = [s.strip() for s in text.split(".") if s.strip()]
        features["sentence_count"] = float(len(sentences))

        # Average sentence length (in words)
        if features["sentence_count"] > 0:
            features["avg_sentence_length"] = features["word_count"] / features["sentence_count"]
        else:
            features["avg_sentence_length"] = 0.0

        # Vocabulary richness (type-token ratio)
        words = self.word_pattern.findall(text.lower())
        if words:
            features["vocabulary_richness"] = len(set(words)) / len(words)
        else:
            features["vocabulary_richness"] = 0.0

        # Character ratio features
        if len(text) > 0:
            features["punctuation_ratio"] = sum(
                1 for c in text if c in ".,;:!?-()[]{}\"'/\\"
            ) / len(text)
            features["uppercase_ratio"] = sum(
                1 for c in text if c.isupper()
            ) / max(len([c for c in text if c.isalpha()]), 1)
            features["digit_ratio"] = sum(
                1 for c in text if c.isdigit()
            ) / len(text)
            features["special_char_ratio"] = sum(
                1 for c in text if not c.isalnum() and not c.isspace()
            ) / len(text)
        else:
            features["punctuation_ratio"] = 0.0
            features["uppercase_ratio"] = 0.0
            features["digit_ratio"] = 0.0
            features["special_char_ratio"] = 0.0

        # Repeated phrase detection
        features["repeated_phrase_count"] = self._count_repeated_phrases(text)

        # Spelling anomaly detection (heuristic-based)
        features["spelling_anomaly_count"] = self._detect_spelling_anomalies(text)

        # Suspicious phrase detection
        features["suspicious_phrase_count"] = float(
            len(self.suspicious_phrase_pattern.findall(text))
        )

        return features

    def _extract_structural_features(
        self, text: str, stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract structural features from document.

        Args:
            text: Document text content.
            stats: Document statistics.

        Returns:
            Dictionary of structural features.
        """
        features: Dict[str, float] = {}

        # Page count from stats or default to 1
        features["page_count"] = float(stats.get("page_count", 1))

        # Paragraph count
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        features["paragraph_count"] = float(len(paragraphs)) if paragraphs else 1.0

        # Table count (look for table-like patterns)
        features["table_count"] = float(stats.get("table_count", 0))
        if features["table_count"] == 0:
            # Heuristic: look for pipe-delimited or tabular content
            table_indicators = text.count("|") + text.count("\t")
            if table_indicators > 5:
                features["table_count"] = 1.0

        # Heading count (lines that look like headings)
        heading_count = 0.0
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) < 80:
                # All caps lines, or lines starting with #, or numbered headings
                if (
                    line.isupper()
                    or re.match(r"^#{1,6}\s", line)
                    or re.match(r"^\d+[\.\)]\s+[A-Z]", line)
                    or re.match(r"^[IVXLC]+[\.\)]\s", line)
                ):
                    heading_count += 1.0
        features["heading_count"] = heading_count

        # Blank page count from stats
        features["blank_page_count"] = float(stats.get("blank_page_count", 0))

        # Font size variance from stats
        features["font_size_variance"] = float(
            stats.get("font_size_variance", 0.0)
        )

        # Alignment variance from stats
        features["alignment_variance"] = float(
            stats.get("alignment_variance", 0.0)
        )

        return features

    def _extract_numerical_features(
        self, text: str, stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract numerical features from document.

        Args:
            text: Document text content.
            stats: Document statistics.

        Returns:
            Dictionary of numerical features.
        """
        features: Dict[str, float] = {}

        # Number count
        numbers = self.number_pattern.findall(text)
        features["number_count"] = float(len(numbers))

        # Monetary values count
        monetary = self.monetary_pattern.findall(text)
        features["monetary_values_count"] = float(len(monetary))

        # Date count
        date_count = 0
        for pattern in self.date_patterns:
            date_count += len(pattern.findall(text))
        features["date_count"] = float(date_count)

        # Total inconsistencies (from stats)
        features["total_inconsistencies"] = float(
            stats.get("total_inconsistencies", 0)
        )

        # ID inconsistencies (from stats)
        features["id_inconsistencies"] = float(
            stats.get("id_inconsistencies", 0)
        )

        return features

    def _extract_ocr_features(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract OCR-related features.

        Args:
            stats: Document statistics.

        Returns:
            Dictionary of OCR features.
        """
        return {
            "ocr_confidence": float(stats.get("ocr_confidence", 1.0)),
            "low_confidence_text_pct": float(
                stats.get("low_confidence_text_pct", 0.0)
            ),
        }

    def _extract_metadata_features(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract metadata features.

        Args:
            metadata: Document metadata.

        Returns:
            Dictionary of metadata features.
        """
        features: Dict[str, float] = {}

        features["has_author"] = 1.0 if metadata.get("author") else 0.0
        features["has_creator"] = 1.0 if metadata.get("creator") else 0.0

        has_dates = bool(
            metadata.get("creation_date") or metadata.get("modification_date")
        )
        features["has_dates"] = 1.0 if has_dates else 0.0

        # Metadata completeness (fraction of expected fields present)
        expected_fields = [
            "title",
            "author",
            "creator",
            "creation_date",
            "modification_date",
            "subject",
            "keywords",
        ]
        present_count = sum(
            1 for field in expected_fields if metadata.get(field)
        )
        features["metadata_completeness"] = present_count / len(expected_fields)

        return features

    def _extract_visual_features(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract visual features.

        Args:
            stats: Document statistics.

        Returns:
            Dictionary of visual features.
        """
        return {
            "image_count": float(stats.get("image_count", 0)),
            "suspicious_region_count": float(
                stats.get("suspicious_region_count", 0)
            ),
        }

    def _count_repeated_phrases(self, text: str, min_words: int = 3) -> float:
        """
        Count repeated multi-word phrases in text.

        Args:
            text: Document text.
            min_words: Minimum word count for a phrase to be considered.

        Returns:
            Count of repeated phrases.
        """
        words = text.lower().split()
        if len(words) < min_words:
            return 0.0

        phrase_counter: Counter = Counter()
        for i in range(len(words) - min_words + 1):
            phrase = " ".join(words[i : i + min_words])
            phrase_counter[phrase] += 1

        # Count phrases that appear more than once
        repeated = sum(
            1 for count in phrase_counter.values() if count > 1
        )
        return float(repeated)

    def _detect_spelling_anomalies(self, text: str) -> float:
        """
        Detect potential spelling anomalies using heuristic methods.

        This is a lightweight heuristic detector, not a full spell checker.
        It looks for patterns that commonly indicate misspellings.

        Args:
            text: Document text.

        Returns:
            Count of potential spelling anomalies.
        """
        anomalies = 0.0
        words = self.word_pattern.findall(text.lower())

        for word in words:
            # Check for doubled letters that shouldn't be doubled
            if re.search(r"([a-z])\1\1", word):
                anomalies += 1.0
            # Check for unusual letter combinations (q not followed by u)
            if "q" in word and "qu" not in word:
                anomalies += 0.5

        return anomalies

    def _extract_dates_from_text(self, text: str) -> List[str]:
        """
        Extract all date strings from text.

        Args:
            text: Document text.

        Returns:
            List of matched date strings.
        """
        dates: List[str] = []
        for pattern in self.date_patterns:
            dates.extend(pattern.findall(text))
        return dates

    def _extract_monetary_values_from_text(self, text: str) -> List[str]:
        """
        Extract all monetary value strings from text.

        Args:
            text: Document text.

        Returns:
            List of matched monetary strings.
        """
        return self.monetary_pattern.findall(text)

    def _extract_ids_from_text(self, text: str) -> List[str]:
        """
        Extract ID-like strings from text.

        Args:
            text: Document text.

        Returns:
            List of matched ID strings.
        """
        ids: List[str] = []
        for pattern in self.id_patterns:
            ids.extend(pattern.findall(text))
        return ids

    def create_feature_vector(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Create a numpy feature vector from document data.

        Args:
            text: Extracted document text content.
            metadata: Document metadata dictionary.
            stats: Document statistics dictionary.

        Returns:
            Numpy array of feature values in defined order.
        """
        features = self.extract_features(text, metadata, stats)
        vector = np.array(
            [features.get(name, 0.0) for name in self.FEATURE_NAMES],
            dtype=np.float64,
        )
        logger.info(
            "Created feature vector of shape %s with %d features",
            vector.shape,
            len(self.FEATURE_NAMES),
        )
        return vector

    def get_feature_names(self) -> List[str]:
        """
        Get the ordered list of feature names.

        Returns:
            List of feature name strings.
        """
        return list(self.FEATURE_NAMES)
