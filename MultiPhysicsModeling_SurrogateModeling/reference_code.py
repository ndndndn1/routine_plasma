"""
Reference implementation inspired by:
  Zhao et al., "Physics insights from a large-scale 2D UEDGE simulation
  database for detachment control in KSTAR", arXiv:2510.16199 (2025).

We mimic the structure of the DivControlNN surrogate: 5 engineering knobs
(upstream density, input power, plasma current, impurity fraction, anomalous
transport) -> detachment indicators (strike-point Te, peak heat flux,
radiation fraction, LFS radiation front position).

The synthetic scalings reproduce the qualitative physics reported in the
paper (Te_sp ~ 3-4 eV at detachment onset, robust to upstream conditions,
with sharp transitions as density or impurity fraction grows). It is NOT
a substitute for UEDGE, just a minimal working example of how to wrap a
multi-physics database in a fast neural surrogate for plasma-state
estimation / control.

Run: python reference_code.py      (needs numpy, torch)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(7)


def simulate_uedge_like(n: int = 8000):
    """Draw knobs uniformly and compute 4 detachment indicators."""
    n_u = RNG.uniform(0.5, 4.0, size=n)        # 1e19 m^-3 upstream sep. density
    P_in = RNG.uniform(1.0, 8.0, size=n)       # MW
    I_p = RNG.uniform(0.4, 1.2, size=n)        # MA
    c_imp = RNG.uniform(0.0, 0.04, size=n)     # impurity fraction
    chi = RNG.uniform(0.2, 2.0, size=n)        # m^2/s anomalous transport

    # Sheath-limited attached regime: Te_sp grows with P_in / (chi * n_u).
    te_attached = 25.0 * (P_in ** 0.6) / (chi ** 0.3 * n_u ** 1.2)
    # Detachment cliff: sharp drop as n_u or c_imp pushes radiation fraction high.
    detach_driver = 0.6 * (n_u / 2.5) + 40.0 * c_imp + 0.05 * (P_in / 5.0) ** (-1)
    detach_frac = 1.0 / (1.0 + np.exp(-6.0 * (detach_driver - 1.0)))
    # Pin Te_sp to the 3-4 eV plateau near detachment onset (paper finding).
    te_sp = te_attached * (1.0 - 0.9 * detach_frac) + (3.0 + 1.0 * RNG.random(n)) * detach_frac
    te_sp = np.clip(te_sp + 0.2 * RNG.standard_normal(n), 1.0, 80.0)

    # Peak heat flux falls as detachment proceeds.
    q_pk = (P_in / (0.4 + 0.3 * I_p)) * (1.0 - 0.85 * detach_frac)
    q_pk = np.clip(q_pk + 0.05 * RNG.standard_normal(n), 0.05, 20.0)

    # Radiation fraction rises with detachment_frac.
    f_rad = np.clip(0.15 + 0.75 * detach_frac + 0.03 * RNG.standard_normal(n), 0.0, 0.98)

    # LFS radiation front position (cm from target), grows as the front moves upstream.
    x_front = 5.0 + 60.0 * detach_frac + 2.0 * RNG.standard_normal(n)
    x_front = np.clip(x_front, 0.0, 90.0)

    X = np.stack([n_u, P_in, I_p, c_imp, chi], axis=-1).astype(np.float32)
    Y = np.stack([te_sp, q_pk, f_rad, x_front], axis=-1).astype(np.float32)
    return X, Y


class DetachmentSurrogate(nn.Module):
    def __init__(self, n_in: int = 5, n_out: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train() -> None:
    X, Y = simulate_uedge_like(n=8000)

    x_mu, x_sd = X.mean(0), X.std(0) + 1e-6
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-6
    Xn = (X - x_mu) / x_sd
    Yn = (Y - y_mu) / y_sd

    n_train = int(0.8 * len(Y))
    idx = RNG.permutation(len(Y))
    tr, te = idx[:n_train], idx[n_train:]

    ds_tr = TensorDataset(torch.from_numpy(Xn[tr]), torch.from_numpy(Yn[tr]))
    dl = DataLoader(ds_tr, batch_size=256, shuffle=True)

    model = DetachmentSurrogate()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(40):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(xb)
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d} train_loss={total/len(ds_tr):.4f}")

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xn[te])).numpy()
    pred = pred_n * y_sd + y_mu
    true = Y[te]
    mae = np.mean(np.abs(pred - true), axis=0)
    labels = ["Te_sp (eV)", "q_pk (MW/m^2)", "f_rad", "x_front (cm)"]
    print("Held-out MAE per output:")
    for lbl, v in zip(labels, mae):
        print(f"  {lbl:20s} {v:.3f}")

    # Sanity check: across the test set, what fraction of "detached" points
    # (f_rad > 0.6) sit in the 3-4 eV Te_sp band the paper emphasises?
    detached = true[:, 2] > 0.6
    in_band = (pred[detached, 0] >= 2.5) & (pred[detached, 0] <= 4.5)
    if detached.sum() > 0:
        frac = 100.0 * in_band.mean()
        print(f"Predicted Te_sp within 2.5-4.5 eV for detached cases: {frac:.1f}%")


if __name__ == "__main__":
    train()
