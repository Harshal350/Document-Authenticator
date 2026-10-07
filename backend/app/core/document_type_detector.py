"""Rule-based document type detection for DocuGuard.

Automatically classifies extracted document text into one of the categories
defined by :class:`app.models.schemas.DocumentType` (invoice, certificate,
identity, educational, ...).  The detected category influences which markers
are *expected* - for example an invoice is expected to carry an invoice
number, dates, seller/buyer information and totals, while a certificate is
expected to name an issuing organisation and a recipient.

Detection is intentionally transparent and explainable: every category scores
itself against weighted keyword/regex patterns and the highest score wins.
Missing fields alone never force a "fake" conclusion - the category is a
supporting signal for the marker engine and the UI.
"""

import re
from typing import Dict, List, Optional, Tuple

from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Keyword / pattern definitions per category (score, pattern)
# --------------------------------------------------------------------------- #
_RULES: Dict[str, List[Tuple[float, str]]] = {
    "invoice": [
        (3.0, r"\binvoice\s*(?:no|number|#|num)\b"),
        (3.0, r"\bbill\s*to\b|\bbilled\s*to\b"),
        (2.5, r"\bsubtotal\b"),
        (2.5, r"\btotal\s*amount\b|\bamount\s*due\b|\bgrand\s*total\b"),
        (2.0, r"\bgst\b|\btax\b|\bvat\b"),
        (2.0, r"\bqty\b|\bquantity\b|\bunit\s*price\b"),
        (1.5, r"\bpurchase\s*order\b|\bpo\s*number\b"),
        (1.0, r"\bshipping\s*address\b"),
    ],
    "receipt": [
        (3.0, r"\breceipt\b"),
        (2.5, r"\bamount\s*paid\b|\bpayment\s*received\b"),
        (2.0, r"\bchange\s*due\b|\bcash\s*tender\b"),
        (2.0, r"\btransaction\s*(?:id|no|number)\b"),
        (1.5, r"\bthank\s*you\s*for\s*your\s*payment\b"),
    ],
    "certificate": [
        (3.0, r"\bcertificate\b"),
        (3.0, r"\bis\s*proudly\s*presented\s*to\b|\bthis\s*is\s*to\s*certify\b"),
        (2.5, r"\baward(ed)?\s*to\b|\bissued\s*to\b"),
        (2.0, r"\bcompletion\b|\bachievement\b|\bmerit\b"),
        (2.0, r"\bregistration\s*no\b|\bcertificate\s*(?:no|number|#)\b"),
        (1.5, r"\bauthorised\s*signatory\b|\bauthorized\s*signature\b"),
    ],
    "identity": [
        (3.0, r"\bdate\s*of\s*birth\b|\bdob\b"),
        (3.0, r"\bpassport\s*(?:no|number|#)\b|\baadhaar\b|\bnational\s*id\b"),
        (2.5, r"\bidentity\b|\bidentification\b"),
        (2.5, r"\bgender\b|\bsex\b.*\bnationality\b"),
        (2.0, r"\bvalid\s*(?:till|until|upto|up\s*to)\b"),
        (1.5, r"\baddress\b.*\bzip\b|\bpostal\s*code\b"),
    ],
    "educational": [
        (3.0, r"\btranscript\b|\bacademic\s*record\b"),
        (2.5, r"\buniversity\b|\bcollege\b|\bschool\b"),
        (2.5, r"\bdegree\b|\bdiploma\b|\bbachelor\b|\bmaster\b"),
        (2.0, r"\bgpa\b|\bgrades?\b|\bmarks?\b obtained\b"),
        (2.0, r"\benrol(?:l)?ment\s*(?:no|number|#)\b|\broll\s*no\b"),
        (1.5, r"\bsession\b|\bsemester\b|\byear\s*of\s*study\b"),
    ],
    "employment": [
        (3.0, r"\bsalary\b|\bctc\b|\bcompensation\b"),
        (2.5, r"\bdesignation\b|\bjob\s*title\b"),
        (2.5, r"\bemployment\s*(?:letter|contract|agreement)\b"),
        (2.0, r"\bdate\s*of\s*joining\b|\bprobation\b"),
        (2.0, r"\bhr\s*department\b|\bhuman\s*resources\b"),
        (1.5, r"\breporting\s*manager\b|\bleave\s*encashment\b"),
    ],
    "financial": [
        (3.0, r"\bbank\s*statement\b|\baccount\s*statement\b"),
        (2.5, r"\baccount\s*(?:no|number|#)\b"),
        (2.5, r"\bopening\s*balance\b|\bclosing\s*balance\b"),
        (2.0, r"\bifsc\b|\bswift\b|\brouting\s*number\b"),
        (2.0, r"\bavailable\s*balance\b|\bdebit\b|\bcredit\b"),
        (1.5, r"\bbranch\b.*\baccount\b"),
    ],
    "application": [
        (3.0, r"\bapplication\s*(?:form|no|number|#)\b"),
        (2.5, r"\bapplicant\b"),
        (2.0, r"\bdeclare\b|\bdeclaration\b"),
        (2.0, r"\bsignature\s*of\s*applicant\b"),
        (1.5, r"\benclosed\b.*\bdocuments?\b"),
    ],
    "agreement": [
        (3.0, r"\bagreement\b"),
        (3.0, r"\bterms\s*and\s*conditions\b"),
        (2.5, r"\bhereby\s*agree\b|\bbetween\s*the\s*(?:parties|undersigned)\b"),
        (2.0, r"\bwitness(?:es)?\b"),
        (2.0, r"\bbinding\b|\bgoverning\s*law\b"),
        (1.5, r"\bin\s*witness\s*whereof\b"),
    ],
    "letter": [
        (2.5, r"\bdear\s+(?:sir|madam|mr|mrs|ms|dr)\b"),
        (2.5, r"\byours\s*(?:sincerely|faithfully|truly)\b"),
        (2.0, r"\bsubject\s*:|\bref\s*:"),
        (1.5, r"\benclosure\b"),
        (1.5, r"\bwith\s*regards\b|\bregards\b"),
    ],
    "government": [
        (3.0, r"\bgovernment\s*of\b|\bgovt\.?\s*of\b"),
        (2.5, r"\bministry\b|\bdepartment\s*of\b|\bdistrict\s*magistrate\b"),
        (2.5, r"\bofficial\s*(?:seal|stamp|copy)\b"),
        (2.0, r"\bunique\b.*\bdocument\b|\baadhaar\b|\bpan\b.*\bcard\b"),
        (2.0, r"\baffidavit\b|\bgazette\b"),
        (1.5, r"\bseal\b.*\bsignature\b"),
    ],
}

_COMPILED: Dict[str, List[Tuple[float, "re.Pattern[str]"]]] = {
    category: [
        (weight, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
        for weight, pattern in patterns
    ]
    for category, patterns in _RULES.items()
}

# Minimum winning score required before a category is accepted.
_MIN_SCORE: float = 3.0


class DocumentTypeDetector:
    """Classify a document into a semantic category from its text."""

    def detect(self, text: str, filename: Optional[str] = None) -> str:
        """Return the most likely document type for ``text``.

        Args:
            text: Extracted document text (OCR or digital).
            filename: Optional original filename; extension/name hints are
                used as a small tie-breaker.

        Returns:
            One of ``invoice``, ``receipt``, ``certificate``, ``identity``,
            ``educational``, ``employment``, ``financial``, ``application``,
            ``agreement``, ``letter``, ``government`` or ``unknown``.
        """
        if not text or not text.strip():
            return "unknown"

        scores: Dict[str, float] = {}
        for category, patterns in _COMPILED.items():
            score = 0.0
            for weight, pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Repeated occurrences add confidence, with diminishing
                    # returns so boilerplate repetition cannot dominate.
                    score += weight * (1.0 + 0.25 * min(len(matches) - 1, 3))
            if score > 0:
                scores[category] = round(score, 2)

        hint = self._filename_hint(filename)
        if hint:
            scores[hint] = scores.get(hint, 0.0) + 2.0

        if not scores:
            return "unknown"

        best_category = max(scores, key=lambda key: scores[key])
        best_score = scores[best_category]
        if best_score < _MIN_SCORE:
            return "unknown"

        runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        confidence_margin = best_score - runner_up
        logger.debug(
            "Document type=%s score=%.2f margin=%.2f candidates=%s",
            best_category,
            best_score,
            confidence_margin,
            scores,
        )
        return best_category

    @staticmethod
    def _filename_hint(filename: Optional[str]) -> Optional[str]:
        """Map simple filename hints onto a document category."""
        if not filename:
            return None
        lowered = filename.lower()
        hints = {
            "invoice": ("invoice", "inv_"),
            "receipt": ("receipt",),
            "certificate": ("certificate", "cert_"),
            "agreement": ("agreement", "contract"),
            "letter": ("letter",),
            "application": ("application",),
            "statement": ("statement",),
        }
        for category, tokens in hints.items():
            if any(token in lowered for token in tokens):
                return "financial" if category == "statement" else category
        return None
