"""Human-readable explanations for DocuGuard risk assessments.

The :class:`ExplainabilityEngine` turns the raw numbers produced by the
:class:`RiskEngine` into plain-English summaries that a non-technical reviewer
can act on.  It deliberately uses cautious phrasing ("a potential issue was
detected") because markers and scores are *indicators of possible manipulation*
rather than definitive proof.

The engine also surfaces the factors that contributed most to the final risk
score (positive and negative), gives a breakdown of the four risk pillars and
summarises both the marker evidence and the machine-learning consensus.
"""

from collections import Counter
from typing import Any, Dict, List

from app.core.risk_engine import (
    DEFAULT_ML_RISK,
    DECISION_LABELS,
    MARKER_WEIGHTED_CAP,
    RISK_WEIGHTS,
    SEVERITY_WEIGHTS,
    RiskEngine,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# A pillar is considered to actively push a document towards "forged" once its
# normalised sub-score crosses these thresholds.
_DIRECTION_HIGH: float = 55.0
_DIRECTION_LOW: float = 25.0

_MAX_TOP_FACTORS: int = 4
_MAX_TOP_MARKERS: int = 3

_NO_MARKERS_TEXT: str = (
    "No forgery indicators were detected during marker analysis."
)
_NO_MODELS_TEXT: str = (
    "No machine-learning model predictions were available, so the ML "
    "component defaulted to a neutral risk level."
)

# Marker categories that are relevant to the consistency checks; only used to
# tailor the wording of the marker summary.
_DATE_FAMILY = {"date", "dates"}
_NUMERICAL_FAMILY = {"numerical", "numbers", "inconsistency", "number"}
_ID_FAMILY = {"id_reference", "id", "reference"}


class ExplainabilityEngine:
    """Generate natural-language explanations for a risk assessment."""

    def __init__(self) -> None:
        """Initialise the engine (and a shared :class:`RiskEngine`)."""
        self.risk_engine = RiskEngine()
        logger.debug("ExplainabilityEngine initialised")

    def explain(
        self,
        risk_score: float,
        decision: str,
        model_predictions: List[Dict[str, Any]],
        markers: List[Dict[str, Any]],
        feature_dict: Dict[str, Any],
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce a complete, human-readable explanation.

        Args:
            risk_score: The overall 0-100 risk score from :class:`RiskEngine`.
            decision: The categorical decision (``original`` / ``suspicious`` /
                ``forged``).
            model_predictions: List of prediction dicts
                (``{model_name, genuine_probability, fake_probability}``).
            markers: List of detected marker dicts
                (``{category, name, severity, score}``).
            feature_dict: Extracted features from :class:`FeatureEngineer`.
            stats: Document statistics from :class:`DocumentProcessor`.

        Returns:
            A dict with ``summary``, ``top_factors``, ``risk_breakdown``,
            ``marker_summary`` and ``ml_summary`` keys.
        """
        if model_predictions is None:
            model_predictions = []
        if markers is None:
            markers = []
        if feature_dict is None:
            feature_dict = {}
        if stats is None:
            stats = {}

        try:
            breakdown = self._compute_breakdown(
                risk_score,
                decision,
                model_predictions,
                markers,
                stats,
                feature_dict,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not rebuild risk breakdown: %s", exc)
            breakdown = self._fallback_breakdown(risk_score, decision)

        top_factors = self._build_top_factors(
            breakdown, model_predictions, markers
        )
        marker_summary = self._summarise_markers(markers)
        ml_summary = self._summarise_ml(model_predictions)
        summary = self._build_summary(
            breakdown, top_factors, marker_summary, ml_summary
        )

        logger.info(
            "Explanation generated for score=%.2f decision=%s "
            "(factors=%d markers=%d models=%d)",
            breakdown.get("risk_score", risk_score),
            breakdown.get("decision", decision),
            len(top_factors),
            len(markers),
            len(model_predictions),
        )
        return {
            "summary": summary,
            "top_factors": top_factors,
            "risk_breakdown": breakdown,
            "marker_summary": marker_summary,
            "ml_summary": ml_summary,
        }

    # ------------------------------------------------------------------ #
    # Risk breakdown
    # ------------------------------------------------------------------ #
    def _compute_breakdown(
        self,
        risk_score: float,
        decision: str,
        model_predictions: List[Dict[str, Any]],
        markers: List[Dict[str, Any]],
        stats: Dict[str, Any],
        feature_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Re-run the risk engine to expose per-pillar sub-scores.

        The caller-provided ``risk_score`` / ``decision`` are the values shown
        in the breakdown; the freshly recomputed sub-scores power the factor
        attribution below.
        """
        computed = self.risk_engine.calculate_risk(
            model_predictions, markers, stats, feature_dict
        )
        decision_key = str(decision).lower()
        return {
            "risk_score": round(_as_float(risk_score, 0.0), 2),
            "decision": decision_key,
            "decision_label": DECISION_LABELS.get(
                decision_key, decision_key
            ),
            "ml_risk_score": self._safe_score(computed.get("ml_risk_score")),
            "marker_risk_score": self._safe_score(
                computed.get("marker_risk_score")
            ),
            "structural_risk_score": self._safe_score(
                computed.get("structural_risk_score")
            ),
            "consistency_risk_score": self._safe_score(
                computed.get("consistency_risk_score")
            ),
            "weights": dict(RISK_WEIGHTS),
            "marker_count": len(markers),
            "model_count": len(model_predictions),
        }

    @staticmethod
    def _safe_score(value: Any) -> float:
        """Coerce a breakdown value to a bounded ``0-100`` score."""
        return round(min(max(_as_float(value, 0.0), 0.0), 100.0), 2)

    @staticmethod
    def _fallback_breakdown(
        risk_score: Any, decision: str
    ) -> Dict[str, Any]:
        """Minimal breakdown used when risk computation itself fails."""
        decision_key = str(decision).lower()
        return {
            "risk_score": round(_as_float(risk_score, 0.0), 2),
            "decision": decision_key,
            "decision_label": DECISION_LABELS.get(
                decision_key, decision_key
            ),
            "ml_risk_score": DEFAULT_ML_RISK,
            "marker_risk_score": 0.0,
            "structural_risk_score": 0.0,
            "consistency_risk_score": 0.0,
            "weights": dict(RISK_WEIGHTS),
            "marker_count": 0,
            "model_count": 0,
        }

    # ------------------------------------------------------------------ #
    # Top factors
    # ------------------------------------------------------------------ #
    def _build_top_factors(
        self,
        breakdown: Dict[str, Any],
        model_predictions: List[Dict[str, Any]],
        markers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank the factors that contributed most to the final risk.

        Each factor keeps a ``factor`` label, an absolute ``contribution``
        (the points this pillar adds to the final 0-100 score) and a
        ``direction`` of ``increase`` / ``decrease`` / ``neutral``.
        """
        pillars = [
            ("ML model predictions", "ml_risk_score", "ml"),
            ("Forensic markers", "marker_risk_score", "marker"),
            ("Structural analysis", "structural_risk_score", "structural"),
            ("Consistency checks", "consistency_risk_score", "consistency"),
        ]

        final_score = float(breakdown.get("risk_score", 0.0))
        factors: List[Dict[str, Any]] = []

        for label, score_key, weight_key in pillars:
            score = self._safe_score(breakdown.get(score_key))
            weight = float(RISK_WEIGHTS.get(weight_key, 0.0))
            contribution = round(weight * score, 2)
            factors.append(
                {
                    "factor": label,
                    "contribution": contribution,
                    "direction": self._factor_direction(score, final_score),
                }
            )

        # The strongest individual markers are equally worth surfacing.
        for marker in self._rank_markers(markers)[: _MAX_TOP_MARKERS]:
            factors.append(
                {
                    "factor": "Marker: {} ({})".format(
                        marker.get("name", "unknown finding"),
                        marker.get("category", "uncategorised"),
                    ),
                    "contribution": self._marker_contribution(marker),
                    "direction": "increase",
                }
            )

        # Sort by absolute contribution so the most impactful factor appears
        # first; ties are broken in favour of risk-increasing factors.
        factors.sort(
            key=lambda f: (
                abs(float(f.get("contribution", 0.0))),
                1.0 if f.get("direction") == "increase" else 0.0,
            ),
            reverse=True,
        )
        return factors[: _MAX_TOP_FACTORS + _MAX_TOP_MARKERS]

    @staticmethod
    def _factor_direction(score: float, final_score: float) -> str:
        """Classify whether a pillar pushes the verdict up or down.

        A very high pillar sub-score actively pulls the document towards
        higher risk; a very low sub-score drags the verdict back down.
        Everything in between is considered neutral.
        """
        if score >= _DIRECTION_HIGH:
            return "increase"
        if score <= _DIRECTION_LOW and final_score >= 30.0:
            return "decrease"
        return "neutral"

    @staticmethod
    def _rank_markers(
        markers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sort markers by their normalised confidence, highest first."""
        ranked = []
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            copy = dict(marker)
            copy["_norm_score"] = _normalise_marker_score(marker)
            ranked.append(copy)
        ranked.sort(key=lambda m: m.get("_norm_score", 0.0), reverse=True)
        for marker in ranked:
            marker.pop("_norm_score", None)
        return ranked

    @staticmethod
    def _marker_contribution(marker: Dict[str, Any]) -> float:
        """Normalised contribution of a single marker to the risk score."""
        norm = _normalise_marker_score(marker)
        severity = str(marker.get("severity") or "low").strip().lower()
        weight = SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["low"])
        return round(
            min(100.0, 100.0 * norm * weight / MARKER_WEIGHTED_CAP), 2
        )

    # ------------------------------------------------------------------ #
    # Natural-language summaries
    # ------------------------------------------------------------------ #
    def _build_summary(
        self,
        breakdown: Dict[str, Any],
        top_factors: List[Dict[str, Any]],
        marker_summary: str,
        ml_summary: str,
    ) -> str:
        """Assemble the one-paragraph explanation."""
        score = float(breakdown.get("risk_score", 0.0))
        label = str(
            breakdown.get("decision_label", breakdown.get("decision", ""))
        )
        increasing = [
            f["factor"]
            for f in top_factors
            if f.get("direction") == "increase"
        ][:3]

        sentence = (
            "The document received a risk score of {:.1f} out of 100 and is "
            "classified as {}.".format(score, label)
        )

        if increasing:
            sentence += " The main risk contributors were {}.".format(
                ", ".join(increasing)
            )
        elif score < 30.0:
            sentence += (
                " No significant risk indicators were detected; the document "
                "looks consistent with an original."
            )
        else:
            sentence += (
                " A balanced combination of signals contributed to this "
                "assessment."
            )

        if marker_summary and marker_summary != _NO_MARKERS_TEXT:
            sentence += " {}".format(marker_summary)
        if ml_summary and ml_summary != _NO_MODELS_TEXT:
            sentence += " {}".format(ml_summary)

        return sentence.strip()

    # ------------------------------------------------------------------ #
    # Marker summary
    # ------------------------------------------------------------------ #
    def _summarise_markers(self, markers: List[Dict[str, Any]]) -> str:
        """Summarise detected markers grouped by category and severity."""
        if not markers:
            return _NO_MARKERS_TEXT

        by_category: Dict[str, int] = Counter()
        by_severity: Dict[str, int] = Counter()
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            by_category[str(marker.get("category") or "uncategorised").lower()] += 1
            by_severity[str(marker.get("severity") or "low").lower()] += 1

        total = sum(by_category.values())
        if total == 0:
            return _NO_MARKERS_TEXT

        category_names = sorted(by_category, key=lambda c: by_category[c], reverse=True)
        category_text = ", ".join(
            "{} ({})".format(name, by_category[name]) for name in category_names
        )

        severity_text = ", ".join(
            "{} {}".format(count, severity)
            for severity, count in sorted(
                by_severity.items(),
                key=lambda kv: SEVERITY_WEIGHTS.get(
                    kv[0], SEVERITY_WEIGHTS["low"]
                ),
                reverse=True,
            )
        )

        # Tailor the phrasing around the consistency-related categories.
        has_date = any(name in _DATE_FAMILY for name in by_category)
        has_numerical = any(name in _NUMERICAL_FAMILY for name in by_category)
        has_id = any(name in _ID_FAMILY for name in by_category)

        consistency_notes = []
        if has_date:
            consistency_notes.append("date-related")
        if has_numerical:
            consistency_notes.append("numerical or arithmetic")
        if has_id:
            consistency_notes.append("reference-number")

        plural = "s" if total != 1 else ""
        were = "were" if total != 1 else "was"
        parts = [
            "A total of {} potential issue{} {} detected".format(
                total, plural, were
            ),
            "across {} categor{}: {}".format(
                len(category_names),
                "ies" if len(category_names) != 1 else "y",
                category_text,
            ),
            "with a severity breakdown of {}".format(severity_text),
        ]
        message = ", ".join(parts) + "."

        if consistency_notes:
            message += (
                " The reported inconsistencies relate to {}.".format(
                    " and ".join(consistency_notes)
                )
            )
        return "Potential issue detected: " + message

    # ------------------------------------------------------------------ #
    # ML summary
    # ------------------------------------------------------------------ #
    def _summarise_ml(self, model_predictions: List[Dict[str, Any]]) -> str:
        """Summarise model consensus or disagreement."""
        if not model_predictions:
            return _NO_MODELS_TEXT

        fake_probs = []
        names = []
        for prediction in model_predictions:
            if not isinstance(prediction, dict):
                continue
            fake = _as_float(prediction.get("fake_probability"), None)
            if fake is None:
                genuine = _as_float(prediction.get("genuine_probability"), None)
                if genuine is not None:
                    fake = 1.0 - min(max(genuine, 0.0), 1.0)
            if fake is not None:
                fake_probs.append(min(max(fake, 0.0), 1.0))
                names.append(str(prediction.get("model_name") or "model"))

        if not fake_probs:
            return _NO_MODELS_TEXT

        highest = max(fake_probs)
        lowest = min(fake_probs)
        average = sum(fake_probs) / len(fake_probs)
        is_plural = len(fake_probs) != 1

        if highest >= 0.60 and lowest >= 0.40:
            verb = (
                "strongly lean{} towards the document being forged".format(
                    "" if is_plural else "s"
                )
                if average >= 0.60
                else "lean{} towards the document being forged".format(
                    "" if is_plural else "s"
                )
            )
        elif highest < 0.40:
            verb = "lean{} towards the document being genuine".format(
                "" if is_plural else "s"
            )
        else:
            verb = (
                "disagree{} with one another, so the models themselves are "
                "uncertain".format("" if is_plural else "s")
            )

        return (
            "Model analysis: {model_count} model{plural} ({models}) {verb} "
            "(average forged probability {avg:.1%}).".format(
                model_count=len(names),
                plural="s" if is_plural else "",
                models=", ".join(names[:3]),
                verb=verb,
                avg=average,
            )
        )


def _as_float(value: Any, default: Any) -> Any:
    """Coerce a value to ``float`` or return ``default`` unchanged."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_marker_score(marker: Dict[str, Any]) -> float:
    """Normalise a marker ``score`` to the ``0-1`` interval."""
    raw = _as_float(marker.get("score", 0.0) or 0.0, 0.0)
    if raw <= 1.0:
        return min(max(raw, 0.0), 1.0)
    return min(max(raw / 100.0, 0.0), 1.0)