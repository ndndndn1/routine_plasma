"""
Reference implementation of the EFIT-mini core idea (Zheng et al., arXiv:2503.19467, 2025).

We reproduce the central architectural choice -- a multi-task MLP that maps a vector
of simulated magnetic probe / flux-loop measurements to (i) scalar plasma parameters,
(ii) a 2D poloidal-flux map psi(R,Z), and (iii) the toroidal current density Jt(R,Z) --
together with a Grad-Shafranov residual regularizer that plays the role of EFIT's
least-squares physical-prior step.

The data here is a synthetic Solov'ev family of equilibria, generated analytically,
because no public dataset accompanies the paper. The point of this script is to
demonstrate the pipeline end-to-end on tiny CPU hardware in a few seconds, not to
match a real tokamak. Replace `make_synthetic_dataset` with a NetCDF/HDF reader to
plug in real EFIT outputs.

Dependencies: numpy, torch.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Synthetic Solov'ev equilibria
# ---------------------------------------------------------------------------
# Solov'ev profiles satisfy Delta* psi = a + b*R^2 with linear p'(psi)=a/mu0,
# ff'(psi)=b. A simple analytical family on a rectangular (R,Z) grid is:
#   psi(R,Z) = 0.5 * R^2 * Z^2 + 0.125 * (R^2 - R0^2)^2  (normalized units)
# We parameterize equilibria by (R0, kappa, delta_amp) and rescale.

GRID_N = 33  # 33x33 keeps the script fast; paper uses 129
R_MIN, R_MAX = 0.5, 1.5
Z_MIN, Z_MAX = -0.6, 0.6
N_PROBES = 32  # synthetic magnetic-probe channels


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    R = torch.linspace(R_MIN, R_MAX, GRID_N)
    Z = torch.linspace(Z_MIN, Z_MAX, GRID_N)
    RR, ZZ = torch.meshgrid(R, Z, indexing="ij")
    return RR, ZZ


def solovev_psi(R0: float, kappa: float, amp: float) -> torch.Tensor:
    RR, ZZ = _grid()
    psi = amp * (0.5 * (RR * ZZ / kappa) ** 2 + 0.125 * (RR**2 - R0**2) ** 2)
    return psi


def grad_shafranov_source(psi: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Right-hand side of Delta* psi = -mu0 R Jt with linear p', ff'."""
    RR, _ = _grid()
    # Jt = R * p'(psi) + ff'(psi) / (mu0 R); for linear profiles -> a*R + b/R.
    return a * RR + b / RR


def magnetic_signals(psi: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Mimic a ring of pickup coils + flux loops by sampling psi and its gradient
    on the rectangular boundary of the (R,Z) box."""
    # Boundary indices (perimeter of the grid).
    top = psi[0, :]
    bot = psi[-1, :]
    left = psi[1:-1, 0]
    right = psi[1:-1, -1]
    perim = torch.cat([top, right, bot.flip(0), left.flip(0)])
    # Subsample to N_PROBES evenly spaced points + small measurement noise.
    idx = torch.linspace(0, perim.numel() - 1, N_PROBES).long()
    sig = perim[idx]
    sig = sig + torch.from_numpy(rng.normal(0.0, 1e-3, size=sig.shape)).float()
    return sig


def scalar_params(psi: torch.Tensor) -> torch.Tensor:
    """Three made-up bulk scalars (stand in for Ip, beta_p, li)."""
    Ip = psi.abs().mean()
    beta_p = psi.std()
    li = psi.max() - psi.min()
    return torch.stack([Ip, beta_p, li])


def make_synthetic_dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, Y_scal, Y_psi, Y_jt = [], [], [], []
    for _ in range(n):
        R0 = float(rng.uniform(0.85, 1.15))
        kappa = float(rng.uniform(1.0, 1.8))
        amp = float(rng.uniform(0.5, 1.5))
        a = float(rng.uniform(0.2, 0.8))
        b = float(rng.uniform(0.1, 0.4))
        psi = solovev_psi(R0, kappa, amp)
        jt = grad_shafranov_source(psi, a, b)
        sig = magnetic_signals(psi, rng)
        X.append(sig)
        Y_scal.append(scalar_params(psi))
        Y_psi.append(psi)
        Y_jt.append(jt)
    return (
        torch.stack(X).float(),
        torch.stack(Y_scal).float(),
        torch.stack(Y_psi).float(),
        torch.stack(Y_jt).float(),
    )


# ---------------------------------------------------------------------------
# 2. Multi-task EFIT-mini network
# ---------------------------------------------------------------------------
class EFITmini(nn.Module):
    def __init__(self, n_in: int = N_PROBES, grid: int = GRID_N, hidden: int = 256):
        super().__init__()
        self.grid = grid
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_scalars = nn.Linear(hidden, 3)
        self.head_psi = nn.Linear(hidden, grid * grid)
        self.head_jt = nn.Linear(hidden, grid * grid)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        s = self.head_scalars(h)
        psi = self.head_psi(h).view(-1, self.grid, self.grid)
        jt = self.head_jt(h).view(-1, self.grid, self.grid)
        return s, psi, jt


# ---------------------------------------------------------------------------
# 3. Grad-Shafranov residual (physical prior, EFIT-style LS step)
# ---------------------------------------------------------------------------
def grad_shafranov_residual(psi: torch.Tensor, jt: torch.Tensor) -> torch.Tensor:
    """Compute Delta* psi + mu0 R Jt on the interior of the grid (finite differences).
    We work in normalized units with mu0 absorbed into Jt's scale."""
    RR, _ = _grid()
    RR = RR.to(psi.device)
    dR = (R_MAX - R_MIN) / (psi.shape[-2] - 1)
    dZ = (Z_MAX - Z_MIN) / (psi.shape[-1] - 1)

    # Delta* psi = R d/dR ( (1/R) dpsi/dR ) + d^2 psi / dZ^2
    psi_R = (psi[:, 2:, 1:-1] - psi[:, :-2, 1:-1]) / (2 * dR)
    psi_RR = (psi[:, 2:, 1:-1] - 2 * psi[:, 1:-1, 1:-1] + psi[:, :-2, 1:-1]) / dR**2
    psi_ZZ = (psi[:, 1:-1, 2:] - 2 * psi[:, 1:-1, 1:-1] + psi[:, 1:-1, :-2]) / dZ**2
    R_int = RR[1:-1, 1:-1]
    delta_star = psi_RR - psi_R / R_int + psi_ZZ
    res = delta_star + R_int * jt[:, 1:-1, 1:-1]
    return res


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def train(seed: int = 0, epochs: int = 30, batch: int = 64, n_train: int = 800,
          n_val: int = 200, lr: float = 2e-3, lambda_gs: float = 1e-2) -> None:
    torch.manual_seed(seed)

    Xtr, Str, Ptr, Jtr = make_synthetic_dataset(n_train, seed=seed)
    Xv, Sv, Pv, Jv = make_synthetic_dataset(n_val, seed=seed + 1)

    loader = DataLoader(TensorDataset(Xtr, Str, Ptr, Jtr),
                        batch_size=batch, shuffle=True)
    model = EFITmini()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, sb, pb, jb in loader:
            opt.zero_grad()
            s_hat, p_hat, j_hat = model(xb)
            loss_s = F.mse_loss(s_hat, sb)
            loss_p = F.mse_loss(p_hat, pb)
            loss_j = F.mse_loss(j_hat, jb)
            # Physical prior: Grad-Shafranov residual on the predicted maps.
            res = grad_shafranov_residual(p_hat, j_hat)
            loss_gs = (res**2).mean()
            loss = loss_s + loss_p + loss_j + lambda_gs * loss_gs
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            s_hat, p_hat, j_hat = model(Xv)
            mae_psi = (p_hat - Pv).abs().mean().item()
            mae_s = (s_hat - Sv).abs().mean().item()
            gs_res = grad_shafranov_residual(p_hat, j_hat).abs().mean().item()
        print(f"epoch {ep + 1:02d}  train_loss={tot / len(Xtr):.4e}  "
              f"val_MAE_psi={mae_psi:.4e}  val_MAE_scalars={mae_s:.4e}  "
              f"GS_residual={gs_res:.4e}")

    # LCFS overlap proxy: fraction of pixels where sign(psi - psi_boundary) agrees.
    with torch.no_grad():
        s_hat, p_hat, j_hat = model(Xv)
        psi_b_true = Pv.amax(dim=(-2, -1), keepdim=True) * 0.95
        psi_b_pred = p_hat.amax(dim=(-2, -1), keepdim=True) * 0.95
        inside_true = (Pv < psi_b_true).float()
        inside_pred = (p_hat < psi_b_pred).float()
        overlap = 1.0 - (inside_true - inside_pred).abs().mean().item()
        print(f"final LCFS pixel-overlap proxy: {overlap * 100:.2f}%")


if __name__ == "__main__":
    train()
