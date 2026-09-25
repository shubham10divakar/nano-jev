import pytest

import nanojev

QUERY = "When was UCL founded?"
PASSAGES = ["UCL was founded in 1826.", "The passage is about London."]


def assert_distribution(probs: dict, options):
    assert list(probs) == list(options)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_relevance_returns_one_distribution_per_passage(decider):
    out = decider.relevance(QUERY, PASSAGES)
    assert len(out) == len(PASSAGES)
    for probs in out:
        assert_distribution(probs, ["irrelevant", "partially relevant", "directly answers"])


def test_sufficient_and_grounded_are_yes_no(decider):
    assert_distribution(decider.sufficient(QUERY, PASSAGES), ["yes", "no"])
    assert_distribution(decider.grounded("UCL was founded in 1900.", PASSAGES[0]), ["yes", "no"])


def test_custom_options(decider):
    options = ["history", "cooking", "football", "music"]
    assert_distribution(decider.decide("Which topic?", options, PASSAGES[0]), options)


def test_option_order_does_not_change_probabilities(decider):
    # Options are scored independently, so reordering them only reorders the output.
    a = decider.decide("Which topic?", ["history", "cooking"], PASSAGES[0])
    b = decider.decide("Which topic?", ["cooking", "history"], PASSAGES[0])
    assert a["history"] == pytest.approx(b["history"], abs=1e-4)


def test_temperature_is_applied(decider):
    saved = dict(decider.temperatures)
    try:
        decider.temperatures["grounded"] = 1000.0  # huge temperature -> near uniform
        probs = decider.grounded("UCL was founded in 1826.", PASSAGES[0])
        assert probs["yes"] == pytest.approx(0.5, abs=0.01)
    finally:
        decider.temperatures.clear()
        decider.temperatures.update(saved)


def test_calibration_and_config_are_loaded(decider):
    assert decider.temperatures["relevance"] == 1.5
    assert decider.version == "0.0-test"
    assert decider.max_length == 128


def test_load_helper(tiny_model_dir):
    d = nanojev.load(str(tiny_model_dir), device="cpu")
    assert d.version == "0.0-test"
