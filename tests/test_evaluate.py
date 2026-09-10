from relay.evaluate import (
    accuracy,
    bootstrap,
    coverage_risk,
    macro_f1,
    round_floats,
    triage_metrics,
)

TRUTH = ["a", "a", "b", "b", "c", "c", "a", "b", "c", "a"]
PRED = ["a", "b", "b", "b", "c", "a", "a", "b", "c", "a"]


def test_accuracy_and_macro_f1():
    assert accuracy(TRUTH, PRED) == 0.8
    assert 0.7 < macro_f1(TRUTH, PRED) < 0.85


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    first = bootstrap(TRUTH, PRED, accuracy)
    assert first == bootstrap(TRUTH, PRED, accuracy)
    assert first[0] <= accuracy(TRUTH, PRED) <= first[1]


def test_triage_counts_missed_escalations():
    should = [True, True, False, False]
    escalated = [True, False, False, True]
    m = triage_metrics(should, escalated)
    assert (m["precision"], m["recall"], m["unsafe_auto_rate"]) == (0.5, 0.5, 0.5)
    assert m["auto_coverage"] == 0.5


def test_coverage_risk_is_monotone_in_the_threshold():
    golden = [
        {"id": 1, "should_escalate": True},
        {"id": 2, "should_escalate": False},
        {"id": 3, "should_escalate": False},
    ]
    outputs = {
        1: {"action": "auto", "confidence": 0.5},
        2: {"action": "auto", "confidence": 0.95},
        3: {"action": "escalate", "confidence": 0.9},
    }
    curve = coverage_risk(golden, outputs)
    coverage = [point["auto_coverage"] for point in curve]
    assert coverage == sorted(coverage, reverse=True)
    assert curve[0]["unsafe_auto_rate"] == 1.0
    assert curve[-1]["unsafe_auto_rate"] == 0.0


def test_round_floats_walks_nested_structures():
    assert round_floats({"a": [1.23456, {"b": 2.34567}]}) == {"a": [1.2346, {"b": 2.3457}]}
