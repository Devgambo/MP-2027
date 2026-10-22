"""Checks for the part 5 mini-surrogate code (no OpenSees runs, no dataset file needed)."""
import numpy as np
import pytest
import torch

from mini_surrogate.data import NT, Normaliser, _features
from mini_surrogate.models import build, n_params
from mini_surrogate.train import metrics


@pytest.mark.parametrize("name", ["fno1d", "fno", "lstm", "bilstm"])
def test_model_shapes_and_size(name):
    m = build(name)
    y = m(torch.randn(2, NT, 5))
    assert y.shape == (2, 20, NT)
    assert 1.0e6 < n_params(m) < 1.6e6         # all four roughly the same size (fair comparison)


def test_lstm_is_causal():
    """Changing the input after step 500 must not change the LSTM output before it."""
    torch.manual_seed(0)
    m = build("lstm").eval()
    x = torch.randn(1, NT, 5)
    x2 = x.clone()
    x2[:, 500:] += 1.0
    with torch.no_grad():
        assert torch.allclose(m(x)[..., :500], m(x2)[..., :500], atol=1e-6)


def test_features():
    acc = np.sin(2 * np.pi * 1.0 * np.arange(5000) * 0.01) * 2.0
    f = _features(acc, 0.5, 1e-4)
    assert f.shape == (NT, 5)
    assert np.all(np.diff(f[:, 2]) >= 0)                     # Arias intensity never decreases
    assert np.allclose(f[:, 3], 0.5) and np.allclose(f[:, 4], -4)
    assert np.abs(f[:, 0]).max() == pytest.approx(2.0 / 9.81, rel=0.02)


def test_normaliser():
    x = torch.randn(10, 50, 5) * 3 + 7
    z = Normaliser(x)(x)
    assert torch.allclose(z.reshape(-1, 5).mean(0), torch.zeros(5), atol=1e-5)
    assert torch.allclose(z.reshape(-1, 5).std(0), torch.ones(5), atol=1e-2)


def test_metrics():
    t = np.arange(NT) * 0.05
    y = np.clip(t / 20, 0, 1)[None, None, :].repeat(3, 0).repeat(4, 1)      # reaches 0.9 at 18 s
    m = metrics(y, y)
    assert m["rmse"] == 0 and m["liq_accuracy"] == 1 and m["liq_onset_mae_s"] == 0
    shifted = np.clip((t - 2) / 20, 0, 1)[None, None, :].repeat(3, 0).repeat(4, 1)
    assert metrics(shifted, y)["liq_onset_mae_s"] == pytest.approx(2.0, abs=0.06)
