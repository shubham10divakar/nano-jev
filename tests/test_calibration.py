import pytest
import torch

pytest.importorskip("sklearn")  # calibration metrics need the `train` extra

from nanojev.calibration import ece, fit_temperature, metrics  # noqa: E402


def test_fit_temperature_recovers_known_temperature():
    # Labels are sampled from softmax(z / 2), so the best temperature for logits z is ~2.
    torch.manual_seed(0)
    z = torch.randn(20000, 3) * 3
    labels = torch.multinomial(torch.softmax(z / 2.0, dim=1), 1).squeeze(1)
    assert fit_temperature(z, labels) == pytest.approx(2.0, rel=0.1)


def test_fit_temperature_falls_back_to_one_on_noise():
    torch.manual_seed(0)
    z = torch.zeros(100, 2)
    labels = torch.randint(0, 2, (100,))
    assert fit_temperature(z, labels) == 1.0


def test_metrics_on_perfect_predictions():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]] * 50)
    labels = torch.tensor([0, 1] * 50)
    m = metrics(logits, labels)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["ece"] < 1e-3 and m["nll"] < 1e-3


def test_ece_of_overconfident_model():
    import numpy as np
    probs = np.array([[0.99, 0.01]] * 100)  # 99% confident, right half the time
    labels = np.array([0, 1] * 50)
    assert ece(probs, labels) == pytest.approx(0.49, abs=0.01)
