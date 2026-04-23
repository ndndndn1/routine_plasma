"""
Reference implementation inspired by:
  Turica et al., "Reconstructions of electron-temperature profiles from the
  EUROfusion Pedestal Database using turbulence models and machine learning",
  arXiv:2504.17486 (2025).

Goal: map electron-density profile + engineering parameters -> electron-
temperature pedestal profile. This is a "virtual-metrology" style inference
for plasma uniformity profiling (spatially resolved mapping over psi_N).

No EUROfusion data is redistributed here. We synthesise a physically
plausible pedestal database using mtanh profiles whose parameters depend on
engineering set-points (Bt, Ip, gas-rate, NBI power, strike-point), then
train a small MLP regressor and evaluate on a held-out 20%.

Usage:
    python reference_code.py
Requires: numpy, torch, matplotlib (optional).
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(0)
N_PSI = 32
PSI = np.linspace(0.85, 1.02, N_PSI)


def mtanh(psi: np.ndarray, height: float, width: float, position: float,
          sep: float, slope_in: float = 0.0) -> np.ndarray:
    """Standard pedestal mtanh fit, following Groebner & Osborne."""
    x = 2.0 * (position - psi) / max(width, 1e-6)
    num = (1.0 + slope_in * x) * np.exp(x) - np.exp(-x)
    den = np.exp(x) + np.exp(-x)
    return sep + 0.5 * (height - sep) * (1.0 + num / den)


def sample_engineering() -> dict:
    return {
        "Bt": RNG.uniform(1.8, 3.5),          # T
        "Ip": RNG.uniform(1.0, 4.0),          # MA
        "P_NBI": RNG.uniform(2.0, 25.0),      # MW
        "gas_rate": RNG.uniform(0.2, 5.0),    # 1e22 e/s
        "strike_cfg": RNG.integers(0, 3),     # {HFS, LFS, V} -> one-hot later
        "delta": RNG.uniform(0.15, 0.45),     # triangularity
    }


def profiles_from_eng(eng: dict) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic but coupled (ne, Te) pedestal profiles."""
    # Density pedestal responds strongly to fuelling and Ip.
    n_ped = 2.0 + 1.2 * (eng["Ip"] / 3.0) + 0.9 * math.log1p(eng["gas_rate"])
    n_width = 0.03 + 0.01 * (eng["gas_rate"] / 3.0)
    n_pos = 0.985 - 0.002 * eng["delta"]
    n_sep = 0.25 * n_ped

    # Te pedestal scales with Bt and Ip, reduced by high gas puff; strike-cfg shifts position.
    T_ped = 0.6 * eng["Bt"] ** 0.7 * eng["Ip"] ** 0.3 * (eng["P_NBI"] / 15.0) ** 0.25
    T_ped *= 1.0 / (1.0 + 0.15 * eng["gas_rate"])
    T_width = 0.025 + 0.008 * (eng["delta"] - 0.3)
    T_pos = 0.985 - 0.004 * eng["strike_cfg"] - 0.003 * eng["delta"]
    T_sep = 0.08 * T_ped

    ne = mtanh(PSI, n_ped, n_width, n_pos, n_sep)
    te = mtanh(PSI, T_ped, T_width, T_pos, T_sep)

    # 3% measurement noise on ne, 6% on Te (as training target).
    ne = ne * (1.0 + 0.03 * RNG.standard_normal(N_PSI))
    te = te * (1.0 + 0.06 * RNG.standard_normal(N_PSI))
    return ne.astype(np.float32), te.astype(np.float32)


def build_dataset(n: int = 4000):
    X_prof, X_eng, Y = [], [], []
    for _ in range(n):
        eng = sample_engineering()
        ne, te = profiles_from_eng(eng)
        strike = np.zeros(3, dtype=np.float32)
        strike[int(eng["strike_cfg"])] = 1.0
        eng_vec = np.array(
            [eng["Bt"], eng["Ip"], eng["P_NBI"], eng["gas_rate"], eng["delta"]],
            dtype=np.float32,
        )
        X_prof.append(ne)
        X_eng.append(np.concatenate([eng_vec, strike]))
        Y.append(te)
    return (
        np.stack(X_prof),
        np.stack(X_eng),
        np.stack(Y),
    )


class PedestalRegressor(nn.Module):
    def __init__(self, n_psi: int = N_PSI, n_eng: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_psi + n_eng, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_psi),
        )

    def forward(self, ne: torch.Tensor, eng: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ne, eng], dim=-1))


def relative_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true) / np.clip(np.abs(true), 1e-3, None)))


def main() -> None:
    X_prof, X_eng, Y = build_dataset(n=4000)

    # Normalise like the paper: per-feature standardisation on engineering
    # parameters, profile-wise scaling by max for shape invariance.
    eng_mu, eng_sd = X_eng.mean(0), X_eng.std(0) + 1e-6
    X_eng_n = (X_eng - eng_mu) / eng_sd
    ne_scale = X_prof.max(axis=1, keepdims=True)
    te_scale = Y.max(axis=1, keepdims=True)
    X_prof_n = X_prof / ne_scale
    Y_n = Y / te_scale

    # 80/20 split.
    n_train = int(0.8 * len(Y))
    idx = RNG.permutation(len(Y))
    tr, te = idx[:n_train], idx[n_train:]

    ds_tr = TensorDataset(
        torch.from_numpy(X_prof_n[tr]),
        torch.from_numpy(X_eng_n[tr]),
        torch.from_numpy(Y_n[tr]),
    )
    ds_te = TensorDataset(
        torch.from_numpy(X_prof_n[te]),
        torch.from_numpy(X_eng_n[te]),
        torch.from_numpy(Y_n[te]),
    )
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)

    model = PedestalRegressor()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        for ne, eng, te_ in dl_tr:
            opt.zero_grad()
            pred = model(ne, eng)
            loss = loss_fn(pred, te_)
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item() * len(te_)
        if epoch % 5 == 0 or epoch == 29:
            print(f"epoch {epoch:3d}  train_loss={epoch_loss / len(ds_tr):.4f}")

    # Evaluate: un-normalise back to physical units and report relative error.
    model.eval()
    with torch.no_grad():
        pred_n = model(
            torch.from_numpy(X_prof_n[te]),
            torch.from_numpy(X_eng_n[te]),
        ).numpy()
    pred = pred_n * te_scale[te]
    true = Y[te]
    err = relative_error(pred, true)
    print(f"held-out mean relative error on Te profile: {err*100:.2f}%")
    within_20 = float(np.mean(np.abs(pred - true) / np.clip(np.abs(true), 1e-3, None) < 0.20))
    print(f"fraction of points within 20%: {within_20*100:.2f}%")


if __name__ == "__main__":
    main()
