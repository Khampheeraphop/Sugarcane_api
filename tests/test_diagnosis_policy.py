from app.domain.diagnosis_policy import decide_screening
from app.domain.evidence import ModelScores

METADATA = {
    "classes": ["EyeSpot", "Healthy", "RedLeafSpot", "RedRot", "RingSpot"],
    "gate_threshold": 0.5,
    "similarity_threshold": 0.35,
    "disease_threshold": 0.5,
}
POLICY = {
    "class_support_thresholds": {
        "EyeSpot": 0.63,
        "Healthy": 0.77,
        "RedLeafSpot": 0.84,
        "RedRot": 0.83,
        "RingSpot": 0.83,
    },
    "requires_confirmation": ["RedRot"],
}


def test_high_softmax_cannot_override_missing_support():
    scores = ModelScores(
        [0.001, 0.005, 0.001, 0.991, 0.002], 0.99, 0.66, [0.55, 0.63, 0.52, 0.66, 0.63]
    )
    result = decide_screening(scores, METADATA, POLICY)
    assert result.status == "uncertain"
    assert result.class_id is None
    assert result.reason == "insufficient_class_support"


def test_leaf_only_red_rot_still_needs_other_plant_parts():
    scores = ModelScores(
        [0.001, 0.005, 0.001, 0.991, 0.002], 0.99, 0.95, [0.55, 0.63, 0.52, 0.95, 0.63]
    )
    result = decide_screening(scores, METADATA, POLICY)
    assert result.status == "uncertain"
    assert result.class_id is None
    assert result.reason == "additional_plant_parts_required"


def test_missing_policy_fails_closed():
    scores = ModelScores([0.99, 0.002, 0.002, 0.002, 0.004], 0.99, 0.95, None)
    assert decide_screening(scores, METADATA, None).status == "uncertain"


def test_supported_healthy_pattern_is_not_forced_into_a_disease():
    scores = ModelScores([0.001, 0.99, 0.002, 0.003, 0.004], 0.99, 0.95, [0.5, 0.9, 0.6, 0.5, 0.6])
    result = decide_screening(scores, METADATA, POLICY)
    assert result.status == "healthy"
    assert result.class_id == "Healthy"
