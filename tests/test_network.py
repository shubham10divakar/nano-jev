"""End-to-end tests against the published weights. Opt in with NANOJEV_NETWORK_TESTS=1.

Every released version must stay listed and loadable, not just the newest one.
"""

import os

import pytest

import nanojev

RELEASED = {"v0.1": "0.1", "v1.0": "1.0"}  # Hub tag -> version in nanojev_config.json

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("NANOJEV_NETWORK_TESTS") != "1",
                       reason="set NANOJEV_NETWORK_TESTS=1 to download the real model"),
]


@pytest.fixture(scope="module", params=sorted(RELEASED))
def real_decider(request):
    return request.param, nanojev.load(request.param, device="cpu")


def test_all_released_versions_are_listed_on_the_hub():
    models, online = nanojev.list_models()
    assert online
    assert set(RELEASED) <= {m.name for m in models}


def test_default_version_is_released():
    assert nanojev.DEFAULT_VERSION in RELEASED


def test_real_model_metadata(real_decider):
    tag, d = real_decider
    assert d.version == RELEASED[tag]
    assert set(d.temperatures) == {"relevance", "sufficient", "grounded"}


def test_real_model_decisions(real_decider):
    _, d = real_decider
    rel = d.relevance("When was UCL founded?", [
        "University College London was founded in 1826.",
        "The Analytical Engine was a proposed mechanical computer.",
    ])
    assert max(rel[0], key=rel[0].get) == "directly answers"
    assert max(rel[1], key=rel[1].get) == "irrelevant"

    ctx = "University College London was founded in 1826."
    assert d.grounded("UCL was founded in 1826.", ctx)["yes"] > 0.5
    assert d.grounded("UCL was founded in 1900.", ctx)["no"] > 0.5
