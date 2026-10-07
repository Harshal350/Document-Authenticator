"""Unified risk scoring engine for DocuGuard.

The :class:`RiskEngine` fuses four independent signals into a single 0-100
risk score that drives the final document decision:

1. **ML risk**          - average forged probability across every model that
   produced a prediction for the analysis (``model_predictions``).
2. **Marker risk**      - severity-weighted sum of every forensic marker
   (``markers``) detected by :class:`MarkerDetector`.
3. **Structural risk**  - layout / OCR / page anomalies derived from the
   extracted ``feature_dict`` and document ``stats``.
4. **Consistency risk** - date, numerical and ID inconsistencies.

The weighted combination (40% ML + 25% markers + 20% structural + 15%
consistency) is mapped onto a categorical decision:

* ``0 - 29``  -> ``original``   ("Likely Genuine")
* ``30 - 59`` -> ``suspicious``
* ``60 - 100``-> ``forged``     ("Likely Fake")

The values returned by :meth:`RiskEngine.calculate_risk` mirror the ``analyses``
table columns (``risk_score``, ``decision``, ``confidence``,
``ml_risk_score``, ``marker_risk_score``, ``structural_risk_score``,
``consistency_risk_score``) so the caller can persist the result verbatim.
"""

from statistics import mean
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Weighting configuration
# --------------------------------------------------------------------------- #
ML_WEIGHT: float = 0.40
MARKER_WEIGHT: float = 0.25
STRUCTURAL_WEIGHT: float = 0.20
CONSISTENCY_WEIGHT: float = 0.15

RISK_WEIGHTS: Dict[str, float] = {
    "ml": ML_WEIGHT,
    "marker": MARKER_WEIGHT,
    "structural": STRUCTURAL_WEIGHT,
    "consistency": CONSISTENCY_WEIGHT,
}

# --------------------------------------------------------------------------- #
# Marker scoring configuration
# --------------------------------------------------------------------------- #
SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

# A single, fully confident high-severity marker contributes 3.0 to the
# weighted sum.  Capping at 6.0 means one such marker lands in the middle of
# the "suspicious" band while multiple strong markers saturate towards 100.
MARKER_WEIGHTED_CAP: float = 6.0

# --------------------------------------------------------------------------- #
# Decision policy (matches the product spec)
# --------------------------------------------------------------------------- #
LOW_RISK_MAX: float = 29.0          # 0-29   -> original
MEDIUM_RISK_MAX: float = 59.0       # 30-59  -> suspicious
HIGH_RISK_MIN: float = 60.0         # 60-100 -> forged

DECISION_LABELS: Dict[str, str] = {
    "original": "original (Likely Genuine)",
    "suspicious": "suspicious",
    "forged": "forged (Likely Fake)",
}

DEFAULT_ML_RISK: float = 50.0       # neutral risk used when no model exists

# Marker categories that contribute to the consistency pillar.  The strings in
# this set are deliberately tolerant of both the :class:`MarkerDetector`
# categories (``date``, ``numerical``, ``id_reference``) and the
# ``MarkerCategory`` enum values (``inconsistency`` and friends).
DATE_CATEGORIES = {"date", "dates"}
NUMERICAL_CATEGORIES = {"numerical", "numbers", "inconsistency", "number"}
ID_CATEGORIES = {"id_reference", "id", "reference"}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` interval."""
    if value < low:
        return low
    if value > high:
        return high
    return value


class RiskEngine:
    """Combines ML predictions, marker evidence, and structural analysis.

    The engine is a pure computation unit: all inputs are passed in as plain
    dictionaries (typically freshly read from the ``model_predictions`` and
    ``markers`` tables or produced in-memory by the detector / feature
    extractor), and the result is a dictionary that matches the ``analyses``
    table schema.
    """

    def __init__(self) -> None:
        """Initialise the engine and cache application settings."""
        self.settings = get_settings()
        logger.debug(
            "RiskEngine initialised (environment=%s)",
            self.settings.environment,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def calculate_risk(
        self,
        model_predictions: List[Dict[str, Any]],
        markers: List[Dict[str, Any]],
        stats: Dict[str, Any],
        feature_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        """Combine all signals into a unified risk profile.

        Args:
            model_predictions: A list of
                ``{model_name, genuine_probability, fake_probability}`` dicts.
                When empty (or when no models are trained yet) the ML pillar
                falls back to a neutral score of ``50``.
            markers: A list of ``{category, name, severity, score}`` dicts as
                produced by :class:`MarkerDetector` / stored in ``markers``.
            stats: Document statistics from :class:`DocumentProcessor`
                (page counts, OCR confidence, table counts, ...).
            feature_dict: Extracted features from :class:`FeatureEngineer`
                (font variance, inconsistency counts, ...).

        Returns:
            A dict with ``risk_score``, ``decision``, ``confidence``,
            ``ml_risk_score``, ``marker_risk_score``,
            ``structural_risk_score`` and ``consistency_risk_score`` keys.
        """
        if model_predictions is None:
            model_predictions = []
        if markers is None:
            markers = []
        if stats is None:
            stats = {}
        if feature_dict is None:
            feature_dict = {}

        try:
            ml_risk_score = self._ml_risk(model_predictions)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ML risk computation failed, using neutral: %s", exc)
            ml_risk_score = DEFAULT_ML_RISK

        try:
            marker_risk_score = self._marker_risk(markers)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Marker risk computation failed, using 0: %s", exc)
            marker_risk_score = 0.0

        try:
            structural_risk_score = self._structural_risk(stats, feature_dict)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Structural risk computation failed, using 0: %s", exc
            )
            structural_risk_score = 0.0

        try:
            consistency_risk_score = self._consistency_risk(
                stats, feature_dict, markers
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Consistency risk computation failed, using 0: %s", exc
            )
            consistency_risk_score = 0.0

        risk_score = round(
            _clamp(
                ML_WEIGHT * ml_risk_score
                + MARKER_WEIGHT * marker_risk_score
                + STRUCTURAL_WEIGHT * structural_risk_score
                + CONSISTENCY_WEIGHT * consistency_risk_score,
                0.0,
                100.0,
            ),
            2,
        )

        decision = self._decision(risk_score)
        confidence = self._confidence(
            model_predictions, markers, marker_risk_score
        )

        result = {
            "risk_score": risk_score,
            "decision": decision,
            "confidence": confidence,
            "ml_risk_score": ml_risk_score,
            "marker_risk_score": marker_risk_score,
            "structural_risk_score": structural_risk_score,
            "consistency_risk_score": consistency_risk_score,
        }

        logger.info(
            "Risk computed: score=%.2f decision=%s confidence=%.2f "
            "(ml=%.2f markers=%.2f structural=%.2f consistency=%.2f)",
            risk_score,
            decision,
            confidence,
            ml_risk_score,
            marker_risk_score,
            structural_risk_score,
            consistency_risk_score,
        )
        return result

    # ------------------------------------------------------------------ #
    # Pillar computations
    # ------------------------------------------------------------------ #
    def _ml_risk(self, model_predictions: List[Dict[str, Any]]) -> float:
        """Average of the fake probabilities across all model predictions.

        Missing ``fake_probability`` values are inferred from the genuine
        probability when possible.  When there are no usable predictions a
        neutral score of ``50`` is returned so the ML pillar never silently
        zeroes out the overall result.
        """
        fake_probabilities: List[float] = []
        for prediction in model_predictions:
            if not isinstance(prediction, dict):
                continue
            fake = self._safe_probability(
                prediction.get("fake_probability"), default=None
            )
            if fake is None:
                genuine = self._safe_probability(
                    prediction.get("genuine_probability"), default=None
                )
                if genuine is not None:
                    fake = _clamp(1.0 - genuine, 0.0, 1.0)
            if fake is not None:
                fake_probabilities.append(fake)

        if not fake_probabilities:
            logger.debug(
                "No usable ML predictions; defaulting ML risk to %.1f",
                DEFAULT_ML_RISK,
            )
            return DEFAULT_ML_RISK

        return round(mean(fake_probabilities) * 100.0, 2)

    def _marker_risk(self, markers: List[Dict[str, Any]]) -> float:
        """Severity-weighted marker score, normalised to 0-100.

        Each marker's confidence (normalised to ``0-1``) is multiplied by its
        severity weight (critical 4x, high 3x, medium 2x, low 1x).  The
        weighted sum is then scaled against :data:`MARKER_WEIGHTED_CAP` so
        that one very strong high-severity marker lands mid-"suspicious" and
        several strong markers saturate towards ``100``.
        """
        detected = 0
        weighted_sum = 0.0
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            detected += 1
            severity = str(marker.get("severity") or "low").strip().lower()
            weight = SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["low"])
            weighted_sum += self._marker_normalised_score(marker) * weight

        if detected == 0:
            return 0.0

        if weighted_sum <= 0.0:
            return 0.0

        return round(min(100.0, 100.0 * weighted_sum / MARKER_WEIGHTED_CAP), 2)

    def _structural_risk(
        self, stats: Dict[str, Any], feature_dict: Dict[str, Any]
    ) -> float:
        """Score structural anomalies (layout, scan quality and pagination).

        The structural pillar blends four sub-signals:

        * layout variance (font sizes + alignment), 40%
        * OCR / scan degradation, 30%
        * page / blank-page anomalies, 20%
        * abnormal table density, 10%
        """
        pages = int(
            self._get_feature(
                feature_dict, stats, ("page_count", "pages"), default=1
            )
        )
        blank_pages = int(
            self._get_feature(
                feature_dict, stats, ("blank_page_count",), default=0
            )
        )
        tables = int(
            self._get_feature(
                feature_dict, stats, ("table_count", "tables_detected"),
                default=0,
            )
        )
        font_variance = self._get_feature(
            feature_dict, stats, ("font_size_variance",), default=0.0
        )
        alignment_variance = self._get_feature(
            feature_dict, stats, ("alignment_variance",), default=0.0
        )
        ocr_confidence = self._get_feature(
            feature_dict, stats, ("ocr_confidence",), default=1.0
        )
        ocr_used = self._get_bool(stats, ("ocr_used",), default=False)
        ocr_pages = int(
            self._get_feature(feature_dict, stats, ("ocr_pages",), default=0)
        )
        image_only_pages = int(
            self._get_feature(
                feature_dict, stats, ("image_only_pages",), default=0
            )
        )

        # Intra-pillar blend weights (they sum to 1.0).
        layout_weight = 0.40
        ocr_weight = 0.30
        page_weight = 0.20
        table_weight = 0.10

        # 1. Layout variance (font sizes weigh more than alignment).
        font_norm = min(1.0, font_variance / 5.0) if font_variance > 0 else 0.0
        align_norm = (
            min(1.0, alignment_variance / 3.0) if alignment_variance > 0 else 0.0
        )
        layout_score = 100.0 * (0.6 * font_norm + 0.4 * align_norm)

        # 2. OCR / scan degradation.
        if ocr_used:
            confidence_penalty = (1.0 - _clamp(ocr_confidence, 0.0, 1.0)) * 100.0
            scan_bump = min(40.0, ocr_pages * 8.0 + image_only_pages * 10.0)
            ocr_score = min(100.0, confidence_penalty + scan_bump)
        else:
            ocr_score = 0.0

        # 3. Page / blank-page anomalies.
        page_anomaly = 0.0
        if pages <= 0:
            page_anomaly = max(page_anomaly, 60.0)
        elif pages >= 40:
            page_anomaly = max(page_anomaly, 30.0)
        if blank_pages > 0 and pages > 0:
            blank_ratio = blank_pages / pages
            page_anomaly = max(
                page_anomaly, min(100.0, blank_ratio * 100.0 * 3.0)
            )

        # 4. Abnormal table density.
        if tables >= 20:
            table_risk = 35.0
        elif tables >= 8:
            table_risk = 20.0
        elif tables >= 3:
            table_risk = 10.0
        else:
            table_risk = 0.0

        structural = (
            layout_weight * layout_score
            + ocr_weight * ocr_score
            + page_weight * page_anomaly
            + table_weight * table_risk
        )
        return round(_clamp(structural, 0.0, 100.0), 2)

    def _consistency_risk(
        self,
        stats: Dict[str, Any],
        feature_dict: Dict[str, Any],
        markers: List[Dict[str, Any]],
    ) -> float:
        """Score date, numerical and reference inconsistencies.

        Explicit inconsistency counters from ``stats`` / ``feature_dict`` are
        merged with the count of marker findings whose category belongs to the
        date / numerical / ID families so the pillar stays meaningful even when
        the detector did not surface dedicated markers.
        """
        date_findings = 0
        numerical_findings = 0
        id_findings = 0

        for marker in markers:
            if not isinstance(marker, dict):
                continue
            category = str(marker.get("category") or "").strip().lower()
            if category in DATE_CATEGORIES:
                date_findings += 1
            elif category in NUMERICAL_CATEGORIES:
                numerical_findings += 1
            elif category in ID_CATEGORIES:
                id_findings += 1

        explicit_inconsistencies = int(
            self._get_feature(
                feature_dict, stats, ("total_inconsistencies",), default=0
            )
        )
        explicit_id_inconsistencies = int(
            self._get_feature(
                feature_dict, stats, ("id_inconsistencies",), default=0
            )
        )

        date_score = min(100.0, date_findings * 30.0)
        numerical_score = min(100.0, numerical_findings * 25.0)
        id_score = min(
            100.0, (id_findings + max(0, explicit_id_inconsistencies)) * 20.0
        )

        category_blend = (
            0.40 * date_score + 0.35 * numerical_score + 0.25 * id_score
        )

        # Any explicit inconsistency counter not already represented by a
        # detected marker contributes an additional, capped bump.
        uncovered = max(
            0, explicit_inconsistencies - (date_findings + numerical_findings)
        )
        uncovered_score = min(40.0, uncovered * 10.0)

        return round(
            _clamp(category_blend + uncovered_score, 0.0, 100.0), 2
        )

    def _decision(self, risk_score: float) -> str:
        """Map a 0-100 risk score onto the categorical decision.

        The configurable thresholds from :mod:`app.config` are preferred so
        the risk engine, the API fallback normaliser and the dashboard risk
        bands always agree; the module-level constants are the fallback.
        """
        medium_min = float(
            getattr(self.settings, "medium_risk_threshold", MEDIUM_RISK_MAX + 1.0)
        )
        high_min = float(getattr(self.settings, "high_risk_threshold", HIGH_RISK_MIN))
        if risk_score < medium_min:
            return "original"
        if risk_score < high_min:
            return "suspicious"
        return "forged"

    def _confidence(
        self,
        model_predictions: List[Dict[str, Any]],
        markers: List[Dict[str, Any]],
        marker_risk_score: float,
    ) -> float:
        """Confidence of the ensemble (0-1).

        With model predictions the confidence is the average of each model's
        peak probability (``max(fake, genuine)``) - the more certain the
        ensembled models are, the higher the confidence.  Without predictions
        a distance-based confidence is derived from how far the marker pillar
        sits from the neutral ``50`` point, discounted by how much marker
        evidence actually exists so sparse evidence never implies certainty.
        """
        certainties: List[float] = []
        for prediction in model_predictions:
            if not isinstance(prediction, dict):
                continue
            fake = self._safe_probability(
                prediction.get("fake_probability"), default=None
            )
            genuine = self._safe_probability(
                prediction.get("genuine_probability"), default=None
            )
            if fake is None and genuine is not None:
                fake = _clamp(1.0 - genuine, 0.0, 1.0)
            if genuine is None and fake is not None:
                genuine = _clamp(1.0 - fake, 0.0, 1.0)
            if fake is None or genuine is None:
                continue
            certainties.append(max(fake, genuine))

        if certainties:
            confidence = mean(certainties)
        else:
            marker_count = sum(
                1 for marker in markers if isinstance(marker, dict)
            )
            evidence_factor = min(1.0, marker_count / 4.0)
            distance = abs(marker_risk_score - 50.0) / 50.0
            confidence = distance * evidence_factor

        return round(_clamp(confidence, 0.0, 1.0), 4)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_probability(
        value: Any, default: Optional[float]
    ) -> Optional[float]:
        """Coerce a value to a probability in ``[0, 1]`` or return default."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return _clamp(number, 0.0, 1.0)

    @staticmethod
    def _marker_normalised_score(marker: Dict[str, Any]) -> float:
        """Normalise a marker score to ``0-1``.

        The :class:`MarkerDetector` emits scores in ``[0, 1]`` while the
        ``markers`` table stores ``[0, 100]``; both are handled transparently.
        """
        try:
            raw = float(marker.get("score", 0.0))
        except (TypeError, ValueError):
            raw = 0.0
        if raw <= 1.0:
            return _clamp(raw, 0.0, 1.0)
        return _clamp(raw / 100.0, 0.0, 1.0)

    @staticmethod
    def _get_feature(
        feature_dict: Dict[str, Any],
        stats: Dict[str, Any],
        keys: tuple,
        default: float = 0.0,
    ) -> float:
        """Return the first available value among preferred keys.

        ``feature_dict`` is preferred over ``stats`` because feature
        dictionaries are the authoritative, normalised source produced by the
        feature extractor; ``stats`` is used as a fallback for keys the
        processor emits under slightly different names.
        """
        for source in (feature_dict, stats):
            for key in keys:
                raw = source.get(key)
                if raw is None:
                    continue
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        return default

    @staticmethod
    def _get_bool(
        source: Dict[str, Any], keys: tuple, default: bool = False
    ) -> bool:
        """Return the first boolean value found among ``keys``."""
        for key in keys:
            raw = source.get(key)
            if raw is not None:
                return bool(raw)
        return default