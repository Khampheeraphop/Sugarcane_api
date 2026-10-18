"""Measurements shared by model runtimes and the decision policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelScores:
    probabilities: list[float]
    gate_score: float
    similarity: float
    class_support: list[float] | None
