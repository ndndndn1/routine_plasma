"""
Reference implementation inspired by:
  Z. Liu et al., "Real-time equilibrium reconstruction by neural network
  based on HL-3 tokamak" (EFITNN), arXiv:2405.11221, 2024.

We reproduce the EFITNN pattern on a synthetic Grad-Shafranov-like dataset:

  inputs  (sparse, ~probe-count) ---> trunk ---> three heads:
    - scalars head:  (Ip, beta_p, li, R_axis, kappa, ...)
    - 2D head:       poloidal flux psi(R, Z) over a 32x32 grid
    - 1D head:       toroidal current density j_phi(rho)

Multi-task learning is the key claim of the paper: training the three
heads jointly should beat training each head alone. The script trains
both regimes and prints the improvement.

Run: python reference_code.py     (needs numpy, torch)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(7)

GRID = 32
N_PROBES = 16
N_LOOPS = 4
N_INPUTS = N_PROBES + N_LOOPS + 2          # + Ip and loop voltage
SCALAR_DIM = 5                              # Ip, beta_p, li, R_axis, kappa
PROFILE_LEN = 32


def random_equilibrium():
    """Sample (Ip, R0, a, kappa, delta) and a Solov'ev-style flux map."""
    Ip = RNG.uniform(0.4, 1.2)
    R0 = RNG.uniform(1.6, 2.0)
    a = RNG.uniform(0.5, 0.8)
    kappa = RNG.uniform(1.2, 1.8)
    delta = RNG.uniform(-0.1, 0.4)
    beta_p = RNG.uniform(0.1, 1.0)
    li = RNG.uniform(0.6, 1.6)

    R = np.linspace(R0 - 1.2 * a, R0 + 1.2 * a, GRID)
    Z = np.linspace(-1.4 * kappa * a, 1.4 * kappa * a, GRID)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    rho2 = ((RR - R0) / a) ** 2 + (ZZ / (kappa * a)) ** 2
    triang = 1.0 + delta * (RR - R0) / a
    psi = Ip * (1.0 - rho2 * triang) * np.exp(-0.2 * rho2)
    psi = psi.astype(np.float32)

    rho = np.linspace(0.0, 1.0, PROFILE_LEN, dtype=np.float32)
    jphi = Ip * (1.0 - rho ** 2) ** (1.5 + 0.5 * (li - 1.0))
    return psi, np.array([Ip, beta_p, li, R0, kappa], dtype=np.float32), jphi


def synth_measurements(psi: np.ndarray) -> np.ndarray:
    """Pseudo magnetic-probe + flux-loop readings sampled from psi."""
    idx = RNG.integers(0, GRID, size=(N_PROBES + N_LOOPS, 2))     # static layout below
    # Use a fixed layout so the inverse problem is consistent across samples:
    rng_fixed = np.random.default_rng(99)
    idx = rng_fixed.integers(0, GRID, size=(N_PROBES + N_LOOPS, 2))
    samples = psi[idx[:, 0], idx[:, 1]]
    # Probes ~ local gradient; flux loops ~ raw psi.
    grad_y, grad_x = np.gradient(psi)
    grad_samples = grad_x[idx[:N_PROBES, 0], idx[:N_PROBES, 1]] + \
                   grad_y[idx[:N_PROBES, 0], idx[:N_PROBES, 1]]
    flux_samples = samples[N_PROBES:]
    return np.concatenate([grad_samples, flux_samples]).astype(np.float32)


def make_dataset(n: int):
    inputs = np.zeros((n, N_INPUTS), dtype=np.float32)
    psi_maps = np.zeros((n, GRID, GRID), dtype=np.float32)
    scalars = np.zeros((n, SCALAR_DIM), dtype=np.float32)
    profiles = np.zeros((n, PROFILE_LEN), dtype=np.float32)
    for i in range(n):
        psi, s, jp = random_equilibrium()
        meas = synth_measurements(psi) + 0.01 * RNG.standard_normal(N_PROBES + N_LOOPS).astype(np.float32)
        Ip = s[0]
        loop_v = float(RNG.normal(0.0, 0.1))
        inputs[i] = np.concatenate([meas, [Ip, loop_v]])
        psi_maps[i] = psi
        scalars[i] = s
        profiles[i] = jp
    return inputs, scalars, psi_maps, profiles


class EFITNN(nn.Module):
    def __init__(self, hidden: int = 256, use_2d: bool = True, use_prof: bool = True):
        super().__init__()
        self.use_2d = use_2d
        self.use_prof = use_prof
        self.trunk = nn.Sequential(
            nn.Linear(N_INPUTS, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_scalar = nn.Linear(hidden, SCALAR_DIM)
        if use_2d:
            base = GRID // 4                               # 8
            self.head_2d_lin = nn.Linear(hidden, base * base * 16)
            self.head_2d_deconv = nn.Sequential(
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.GELU(),
                nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
            )
            self._base = base
        if use_prof:
            self.head_prof = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, PROFILE_LEN),
            )

    def forward(self, x):
        h = self.trunk(x)
        s = self.head_scalar(h)
        p2d = None
        prof = None
        if self.use_2d:
            b = self.head_2d_lin(h).view(-1, 16, self._base, self._base)
            p2d = self.head_2d_deconv(b).squeeze(1)
        if self.use_prof:
            prof = self.head_prof(h)
        return s, p2d, prof


def train(model: EFITNN, data, epochs: int = 12, w2d: float = 1.0, wprof: float = 1.0):
    inputs, scalars, psi_maps, profiles = data
    ds = TensorDataset(
        torch.from_numpy(inputs), torch.from_numpy(scalars),
        torch.from_numpy(psi_maps), torch.from_numpy(profiles),
    )
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        total = 0.0
        for xb, sb, mb, pb in dl:
            opt.zero_grad()
            s_hat, m_hat, p_hat = model(xb)
            loss = nn.functional.mse_loss(s_hat, sb)
            if m_hat is not None:
                loss = loss + w2d * nn.functional.mse_loss(m_hat, mb)
            if p_hat is not None:
                loss = loss + wprof * nn.functional.mse_loss(p_hat, pb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(xb)
        if epoch % 3 == 0:
            print(f"[efitnn] epoch {epoch:2d}  loss={total/len(ds):.4f}")
    return model


def scalar_mae(model: EFITNN, data) -> float:
    inputs, scalars, _, _ = data
    with torch.no_grad():
        s_hat, _, _ = model(torch.from_numpy(inputs))
    return float(np.mean(np.abs(s_hat.numpy() - scalars)))


def main() -> None:
    train_data = make_dataset(2000)
    test_data = make_dataset(400)

    print("[multi-task: scalars + flux map + current profile]")
    multi = EFITNN(use_2d=True, use_prof=True)
    train(multi, train_data)
    mt_mae = scalar_mae(multi, test_data)
    print(f"scalar MAE = {mt_mae:.4f}")

    print("[single-task: scalars only]")
    single = EFITNN(use_2d=False, use_prof=False)
    train(single, train_data)
    st_mae = scalar_mae(single, test_data)
    print(f"scalar MAE = {st_mae:.4f}")

    print(f"\nmulti-task improvement on scalar prediction: "
          f"{(st_mae - mt_mae) / st_mae * 100:.1f} %")


if __name__ == "__main__":
    main()
