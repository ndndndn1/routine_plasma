"""
Reference implementation inspired by:
  "A microwave reflection diagnostics of inhomogeneous plasma distribution
  based on multi-peak points phenomenon", Phys. Plasmas 32, 042113 (2025).

We replicate the structure of the paper:

  1. Stratified-slab forward model:  given a discretised n_e(x) and nu_en(x)
     in N layers, compute the complex reflection coefficient Gamma(f) of a
     normally-incident plane wave via the transfer-matrix method using the
     cold-plasma dielectric function in each layer.
  2. Generate |Gamma(f)| spectra for random Gaussian density profiles.
  3. Locate the local maxima of |Gamma(f)| -> a sparse vector of peak
     frequencies.
  4. Train an MLP (peak positions -> profile parameters n_peak, x0, width,
     nu_en) and report held-out error.

Run: python reference_code.py     (needs numpy, torch)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(13)

C0 = 2.99792458e8
EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31

SLAB_THICK = 0.05                       # 5 cm slab
N_LAYERS = 80
FREQS_HZ = np.linspace(0.5e9, 30.0e9, 256)
N_PEAKS_INPUT = 6


def cold_plasma_eps(omega, n_e, nu_en):
    omega_p2 = n_e * QE ** 2 / (EPS0 * ME)
    return 1.0 - omega_p2 / (omega * (omega + 1j * nu_en))


def profile(n_peak, x0, width, nu_en, n_layers=N_LAYERS, thick=SLAB_THICK):
    x = np.linspace(0.0, thick, n_layers)
    n_e = n_peak * np.exp(-0.5 * ((x - x0) / width) ** 2)
    nu = np.full_like(n_e, nu_en)
    return n_e, nu, x


def reflection_spectrum(n_e_layers, nu_layers, freqs=FREQS_HZ, thick=SLAB_THICK):
    """Transfer-matrix reflection of a stratified cold plasma slab."""
    omegas = 2.0 * np.pi * freqs
    d = thick / len(n_e_layers)
    gamma = np.zeros(len(omegas), dtype=complex)
    for i, omega in enumerate(omegas):
        # Start from the back side in vacuum.
        Z_load = 377.0
        for n_e, nu in zip(n_e_layers[::-1], nu_layers[::-1]):
            eps = cold_plasma_eps(omega, n_e, nu)
            k = omega / C0 * np.sqrt(eps + 0j)
            Z_l = 377.0 / np.sqrt(eps + 0j)
            tan_kd = np.tan(k * d)
            Z_load = Z_l * (Z_load + 1j * Z_l * tan_kd) / (Z_l + 1j * Z_load * tan_kd)
        gamma[i] = (Z_load - 377.0) / (Z_load + 377.0)
    return gamma


def find_peaks_positions(spec_abs, k: int = N_PEAKS_INPUT):
    # Local maxima with simple neighborhood comparison.
    peaks = []
    for i in range(2, len(spec_abs) - 2):
        if spec_abs[i] > spec_abs[i - 1] and spec_abs[i] > spec_abs[i + 1]:
            peaks.append((spec_abs[i], i))
    peaks.sort(reverse=True)
    idx = sorted([p[1] for p in peaks[:k]])
    # Pad if fewer than k peaks found.
    while len(idx) < k:
        idx.append(idx[-1] if idx else len(spec_abs) // 2)
    return np.array([FREQS_HZ[i] for i in idx], dtype=np.float32)


def sample_profile():
    n_peak = 10.0 ** RNG.uniform(17.0, 19.0)
    x0 = RNG.uniform(0.01, SLAB_THICK - 0.01)
    width = RNG.uniform(0.003, 0.015)
    nu_en = 10.0 ** RNG.uniform(7.0, 9.0)
    return n_peak, x0, width, nu_en


def make_dataset(n: int):
    features = np.zeros((n, N_PEAKS_INPUT), dtype=np.float32)
    targets = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        n_peak, x0, width, nu_en = sample_profile()
        n_e_layers, nu_layers, _ = profile(n_peak, x0, width, nu_en)
        spec = np.abs(reflection_spectrum(n_e_layers, nu_layers))
        feats = find_peaks_positions(spec) / 1e9
        features[i] = feats
        targets[i] = np.array([np.log10(n_peak), x0, width, np.log10(nu_en)],
                              dtype=np.float32)
    return features, targets


class InverseMLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_PEAKS_INPUT, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    print("generating synthetic training data (transfer-matrix forward model)...")
    x_train, y_train = make_dataset(600)
    x_test, y_test = make_dataset(100)

    x_mean = x_train.mean(axis=0); x_std = x_train.std(axis=0) + 1e-6
    y_mean = y_train.mean(axis=0); y_std = y_train.std(axis=0) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32)
    y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = InverseMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    for epoch in range(80):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.detach().item() * len(xb)
        sched.step()
        if epoch % 16 == 0:
            print(f"[mw-inv] epoch {epoch:2d}  mse={total/len(ds):.4f}")

    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_test_n)).numpy()
    pred = pred_n * y_std + y_mean
    log_n_err = np.abs(pred[:, 0] - y_test[:, 0]).mean()
    x0_err = np.abs(pred[:, 1] - y_test[:, 1]).mean() * 1e3
    w_err = np.abs(pred[:, 2] - y_test[:, 2]).mean() * 1e3
    log_nu_err = np.abs(pred[:, 3] - y_test[:, 3]).mean()
    print(f"\nheld-out MAE:")
    print(f"  log10(n_peak)  = {log_n_err:.3f} dex   (range ~ 2 dex)")
    print(f"  x0 (profile center) = {x0_err:.2f} mm  (range ~ 30 mm)")
    print(f"  width             = {w_err:.2f} mm     (range ~ 12 mm)")
    print(f"  log10(nu_en)    = {log_nu_err:.3f} dex   (range ~ 2 dex)")


if __name__ == "__main__":
    main()
