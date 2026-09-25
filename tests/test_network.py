"""End-to-end tests against the published weights. Opt in with NANOJEV_NETWORK_TESTS=1."""

import os

import pytest

import nanojev

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("NANOJEV_NETWORK_TESTS") != "1",
                       reason="set NANOJEV_NETWORK_TESTS=1 to download the real model"),
]


@pytest.fixture(scope="module")
def real_decider():
    return nanojev.load("v0.1", device="cpu")


def test_v01_is_listed_on_the_hub():
    models, online = nanojev.list_models()
    assert online
    assert "v0.1" in [m.name for m in models]


def test_real_model_metadata(real_decider):
    assert real_decider.version == "0.1"
    assert set(real_decider.temperatures) == {"relevance", "sufficient", "grounded"}


def test_real_model_decisions(real_decider):
    rel = real_decider.relevance("When was UCL founded?", [
        "University College London was founded in 1826.",
        "The Analytical Engine was a proposed mechanical computer.",
    ])
    assert max(rel[0], key=rel[0].get) == "directly answers"
    assert max(rel[1], key=rel[1].get) == "irrelevant"

    ctx = "University College London was founded in 1826."
    assert real_decider.grounded("UCL was founded in 1826.", ctx)["yes"] > 0.5
    assert real_decider.grounded("UCL was founded in 1900.", ctx)["no"] > 0.5
