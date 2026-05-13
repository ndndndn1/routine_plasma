"""
Reference implementation of the core idea in Stokolesov et al.,
"Reconstructing the Plasma Boundary with a Reduced Set of Diagnostics"
(arXiv:2505.10709, May 2025).

The paper's main empirical result is that an MLP can recover the last closed flux
surface (LCFS) of a DIII-D plasma from PF coil currents alone, and that adding plasma
current Ip and loop voltage Vloop reduces the mean point displacement by ~25%.

Since the DIII-D dataset is not public, we reproduce the *experimental design* on a
synthetic but physically motivated dataset:
  - Each "shot" has a randomly drawn LCFS parameterized by (R0, a, kappa, delta, ZS),
    i.e., major radius, minor radius, elongation, triangularity, Shafranov shift.
  - The LCFS is discretized at 64 fixed poloidal angles.
  - We simulate ~20 PF coil currents via a fixed linear "diagnostic" matrix that maps
    the shape parameters to coil currents (a stand-in for the Biot-Savart forward map).
  - We add Ip and Vloop as two extra channels for the augmented model.

The script trains the two MLPs in the paper (Model A: coils only; Model B: coils+Ip+Vloop)
and prints the mean point displacement on a held-out set.

Dependencies: numpy, torch.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


N_POINTS = 64    # boundary discretization
N_COILS = 20     # PF coil channels
SEED = 0


# ---------------------------------------------------------------------------
# 1. Synthetic LCFS family
# ---------------------------------------------------------------------------
def lcfs_contour(R0: float, a: float, kappa: float, delta: float, ZS: float) -> torch.Tensor:
    """Miller-style boundary parameterization."""
    theta = torch.linspace(0, 2 * math.pi, N_POINTS + 1)[:-1]
    R = R0 + a * torch.cos(theta + delta * torch.sin(theta))
    Z = ZS + kappa * a * torch.sin(theta)
    return torch.stack([R, Z], dim=-1)  # (N_POINTS, 2)


def sample_shot(rng: np.random.Generator) -> tuple[torch.Tensor, dict]:
    params = dict(
        R0=float(rng.uniform(1.6, 1.8)),     # DIII-D R0 ~ 1.67 m
        a=float(rng.uniform(0.55, 0.65)),    # minor radius
        kappa=float(rng.uniform(1.4, 2.0)),  # elongation
        delta=float(rng.uniform(-0.5, 0.7)), # triangularity (NT and PT both)
        ZS=float(rng.uniform(-0.1, 0.1)),    # vertical shift
        Ip=float(rng.uniform(0.6, 1.5)),     # MA
        Vloop=float(rng.uniform(0.1, 1.0)),  # V
    )
    return lcfs_contour(params["R0"], params["a"], params["kappa"],
                        params["delta"], params["ZS"]), params


# Fixed linear "forward diagnostic" mapping shape -> coil currents.
# In reality this would be a Biot-Savart-style sparse linear map; here we pick a
# deterministic random matrix and add measurement noise.
def build_forward_matrix(seed: int = 1234) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    # Inputs: 5 shape params (R0, a, kappa, delta, ZS).
    # Outputs: N_COILS coil currents.
    A = torch.from_numpy(rng.normal(0.0, 1.0, size=(N_COILS, 5))).float()
    return A


FWD = build_forward_matrix()


def coil_currents(params: dict, noise: float = 0.01) -> torch.Tensor:
    shape_vec = torch.tensor([params["R0"], params["a"], params["kappa"],
                              params["delta"], params["ZS"]]).float()
    coils = FWD @ shape_vec
    coils = coils + torch.randn_like(coils) * noise
    return coils


def make_dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_a, X_b, Y = [], [], []
    for _ in range(n):
        boundary, params = sample_shot(rng)
        coils = coil_currents(params)
        x_a = coils
        x_b = torch.cat([coils, torch.tensor([params["Ip"], params["Vloop"]]).float()])
        X_a.append(x_a)
        X_b.append(x_b)
        Y.append(boundary.flatten())
    return (torch.stack(X_a).float(), torch.stack(X_b).float(),
            torch.stack(Y).float())


# ---------------------------------------------------------------------------
# 2. MLP
# ---------------------------------------------------------------------------
class BoundaryMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = 2 * N_POINTS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Mean point displacement (paper's metric)
# ---------------------------------------------------------------------------
def mean_point_displacement(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1, N_POINTS, 2)
    target = target.view(-1, N_POINTS, 2)
    d = (pred - target).pow(2).sum(dim=-1).sqrt()  # per-point distance (m)
    return d.mean()


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def train_one(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
              Xv: torch.Tensor, Yv: torch.Tensor, epochs: int = 40,
              batch: int = 128, lr: float = 2e-3, tag: str = "model") -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                mpd = mean_point_displacement(model(Xv), Yv).item()
            print(f"  [{tag}] epoch {ep + 1:02d}  val_MPD={mpd:.4f} m")
    model.eval()
    with torch.no_grad():
        return mean_point_displacement(model(Xv), Yv).item()


def main():
    torch.manual_seed(SEED)
    Xa_tr, Xb_tr, Y_tr = make_dataset(4000, seed=SEED)
    Xa_v, Xb_v, Y_v = make_dataset(1000, seed=SEED + 1)

    print("Model A: coil currents only")
    model_a = BoundaryMLP(n_in=N_COILS)
    mpd_a = train_one(model_a, Xa_tr, Y_tr, Xa_v, Y_v, tag="A")

    print("Model B: coil currents + Ip + Vloop")
    model_b = BoundaryMLP(n_in=N_COILS + 2)
    mpd_b = train_one(model_b, Xb_tr, Y_tr, Xb_v, Y_v, tag="B")

    print()
    print(f"Final mean point displacement -- coils only:        {mpd_a:.4f} m")
    print(f"Final mean point displacement -- coils + Ip+Vloop:  {mpd_b:.4f} m")
    print(f"Relative improvement from adding Ip+Vloop:          "
          f"{100 * (mpd_a - mpd_b) / mpd_a:.1f}%")


if __name__ == "__main__":
    main()
