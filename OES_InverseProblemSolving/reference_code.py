"""
Reference implementation inspired by:
  Y.-F. Wang, X.-M. Zhu, "Development of optical emission spectroscopy method
  with neural network model: Case study of determining the electron density
  in a xenon microwave discharge", J. Appl. Phys. 136, 243302 (2024).

The paper builds an OES inverse model where a neural network takes selected
emission-line ratios as input and outputs the electron density n_e, while
absorbing instrument noise + atomic-data uncertainty through training-time
data augmentation.

Here we mirror the structure with a toy collisional-radiative-like forward
model so the script is self contained:
  1. Forward CR-like model maps (n_e, T_e) -> intensities of 4 fictitious
     "line states" via excitation/de-excitation balance and quenching.
  2. Instrument-disturbance model multiplies each line by a per-line random
     gain (calibration drift) and adds additive Gaussian noise.
  3. An MLP maps the 3 line ratios (using line #1 as reference) -> log10(n_e).

Run: python reference_code.py     (needs numpy, torch)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(0)

N_LINES = 4
LINE_RATIOS = N_LINES - 1
LOG_NE_RANGE = (16.0, 19.0)           # 10^16 .. 10^19 m^-3
TE_RANGE_EV = (1.0, 5.0)


def cr_intensities(log_ne: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Toy CR-like map.

    Each "line" k has a direct-excitation rate (~ n_e) and a stepwise
    excitation contribution from a metastable buffer (~ n_e^2 in the
    coronal limit), with different relative weights per line. This breaks
    the degeneracy of line ratios under a pure n_e prefactor and gives the
    ratios genuine n_e dependence, which is the practical reason CR-model
    line-ratio inversion can recover n_e at all.
    """
    ne = 10.0 ** log_ne
    e_thresh = np.array([2.0, 4.0, 6.0, 8.0])                   # eV
    deexc = np.array([1.0, 1.5, 2.2, 3.0])                      # arb.
    # direct (a_k) vs stepwise (b_k) excitation weights, differ per line.
    a_k = np.array([1.0, 0.6, 0.2, 0.05])
    b_k = np.array([0.05, 0.4, 1.0, 1.6])
    te_b = te[..., None]
    ne_b = ne[..., None]
    excite = np.exp(-e_thresh / np.clip(te_b, 1e-2, None))
    quench = 1.0 / (1.0 + deexc * ne_b / 1e18)
    n_norm = ne_b / 1e18
    intensity = (a_k * ne_b + b_k * ne_b * n_norm) * excite * quench
    return intensity.astype(np.float32)


def disturb_lines(intensity: np.ndarray, gain_sigma: float = 0.10,
                  noise_sigma: float = 0.03) -> np.ndarray:
    """Per-line multiplicative calibration drift + additive Gaussian noise."""
    gains = np.exp(gain_sigma * RNG.standard_normal(intensity.shape))
    out = intensity * gains
    out = out + noise_sigma * out.mean(axis=-1, keepdims=True) * \
        RNG.standard_normal(out.shape)
    return np.clip(out, 1e-8, None).astype(np.float32)


def line_ratios(intensity: np.ndarray) -> np.ndarray:
    """Use line[0] as reference; report log ratios of remaining lines."""
    ref = intensity[..., :1]
    ratios = intensity[..., 1:] / np.clip(ref, 1e-12, None)
    return np.log10(ratios).astype(np.float32)


def make_dataset(n: int, disturb: bool):
    log_ne = RNG.uniform(*LOG_NE_RANGE, size=(n,)).astype(np.float32)
    te = RNG.uniform(*TE_RANGE_EV, size=(n,)).astype(np.float32)
    intensity = cr_intensities(log_ne, te)
    if disturb:
        intensity = disturb_lines(intensity)
    x = line_ratios(intensity)
    # The network jointly predicts (log_ne, T_e): treating T_e as a nuisance
    # parameter we marginalize over creates an inverse-problem ambiguity that
    # the line-ratio features alone cannot break. Predicting both removes it.
    y = np.stack([log_ne, te], axis=-1)
    return x, y


class InverseMLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LINE_RATIOS, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train(model: InverseMLP, x: np.ndarray, y: np.ndarray, epochs: int = 60):
    # Z-normalize targets so log_ne and T_e are on comparable scales.
    y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (y.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    y_norm = (y - y_mean) / y_std
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_norm))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = nn.HuberLoss()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = huber(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(xb)
        sched.step()
        if epoch % 10 == 0:
            print(f"[oes-mlp] epoch {epoch:2d}  huber={total/len(ds):.4f}")
    model._y_mean = torch.from_numpy(y_mean)
    model._y_std = torch.from_numpy(y_std)
    return model


def evaluate(model: InverseMLP, x: np.ndarray, y: np.ndarray) -> dict:
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x))
        pred = (pred_norm * model._y_std + model._y_mean).numpy()
    log_ne_mae = np.abs(pred[:, 0] - y[:, 0]).mean()
    te_mae = np.abs(pred[:, 1] - y[:, 1]).mean()
    # Median relative ne error is more robust than the mean of (10^|err|-1)
    # because that quantity has a heavy tail.
    rel_err = np.median(10.0 ** np.abs(pred[:, 0] - y[:, 0]) - 1.0)
    return {"log_ne_mae": float(log_ne_mae), "te_mae": float(te_mae),
            "median_rel": float(rel_err)}


def main() -> None:
    # Training: include instrument disturbance as data augmentation.
    x_train, y_train = make_dataset(n=20000, disturb=True)
    model = InverseMLP()
    train(model, x_train, y_train)

    # Two test sets: clean CR vs noisy CR.
    x_clean, y_clean = make_dataset(n=4000, disturb=False)
    x_noisy, y_noisy = make_dataset(n=4000, disturb=True)
    clean = evaluate(model, x_clean, y_clean)
    noisy = evaluate(model, x_noisy, y_noisy)
    print(f"clean test  log10(ne) MAE = {clean['log_ne_mae']:.3f} dex  "
          f"Te MAE = {clean['te_mae']:.2f} eV  "
          f"median relative ne error = {clean['median_rel']*100:.1f} %")
    print(f"noisy test  log10(ne) MAE = {noisy['log_ne_mae']:.3f} dex  "
          f"Te MAE = {noisy['te_mae']:.2f} eV  "
          f"median relative ne error = {noisy['median_rel']*100:.1f} %")


if __name__ == "__main__":
    main()
