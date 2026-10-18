"""Pure decision rules. A classifier score alone never establishes a diagnosis."""

from dataclasses import dataclass
from typing import Literal

DecisionStatus = Literal["unrecognized", "uncertain", "healthy", "suspected"]


@dataclass(frozen=True)
class ScreeningDecision:
    status: DecisionStatus
    reason: str
    class_id: str | None = None


def decide_screening(scores, metadata: dict, policy: dict | None) -> ScreeningDecision:
    if (
        scores.gate_score < metadata["gate_threshold"]
        or scores.similarity < metadata["similarity_threshold"]
    ):
        return ScreeningDecision("unrecognized", "outside_sugarcane_reference")
    if policy is None or scores.class_support is None:
        return ScreeningDecision("uncertain", "support_policy_unavailable")
    index = max(range(len(scores.probabilities)), key=scores.probabilities.__getitem__)
    class_id = metadata["classes"][index]
    if scores.probabilities[index] < metadata["disease_threshold"]:
        return ScreeningDecision("uncertain", "insufficient_classifier_evidence")
    if scores.class_support[index] < policy["class_support_thresholds"][class_id]:
        return ScreeningDecision("uncertain", "insufficient_class_support")
    if class_id in policy["requires_confirmation"]:
        return ScreeningDecision("uncertain", "additional_plant_parts_required")
    return ScreeningDecision(
        "healthy" if class_id == "Healthy" else "suspected", "supported_pattern", class_id
    )
