"""
Reference implementation inspired by:
  "Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum
  Variable Capacitor Positions of the Impedance Matching Unit",
  MDPI Electronics 14(10), 2022 (2025).

Idea:  in a 50 ohm-matched L-network, the two VVC capacitances are a
one-to-one function of the plasma load impedance Z_p = R_p + j X_p, so
reading the VVC encoder positions is enough to estimate Z_p without any
external V-I probe (a "virtual metrology" of plasma impedance).

We implement this with two ingredients:
  1. Forward L-network model:  Z_in(C_load, C_tune; Z_p) of the matching
     network (shunt C_load - series L_s - shunt C_tune - plasma load).
  2. For each (R_p, X_p) on a synthetic process grid, numerically search
     (C_load, C_tune) that make Z_in = 50 + 0j.  This produces a paired
     dataset (C_load, C_tune) <-> (R_p, X_p).
  3. Inverse model:  fit an MLP that maps (C_load, C_tune) -> (R_p, X_p),
     and evaluate the noise floor when VVC readouts are quantised by the
     motor-encoder step.

Run: python reference_code.py     (needs numpy, torch)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(3)

F = 13.56e6
OMEGA = 2.0 * np.pi * F
L_SERIES = 0.5e-6                        # fixed 0.5 uH series inductor
Z0 = 50.0


def network_Zin(C_load, C_tune, R_p, X_p, omega=OMEGA, L_s=L_SERIES):
    """Source-side input impedance of a shunt-series-shunt L-network.

    Source --[shunt C_load]-- [series L_s] --[shunt C_tune]-- [Z_p]
    """
    Z_p = R_p + 1j * X_p
    Y_tune = 1j * omega * C_tune
    Z_after_tune = 1.0 / (Y_tune + 1.0 / Z_p)
    Z_after_L = 1j * omega * L_s + Z_after_tune
    Y_load = 1j * omega * C_load
    Z_in = 1.0 / (Y_load + 1.0 / Z_after_L)
    return Z_in


def _seed_from_grid(R_p, X_p, omega=OMEGA):
    grid = np.linspace(20e-12, 3000e-12, 24)
    best_C_load = 200e-12; best_C_tune = 400e-12; best_err = np.inf
    for c_l in grid:
        for c_t in grid:
            Z = network_Zin(c_l, c_t, R_p, X_p, omega=omega)
            err = (Z.real - Z0) ** 2 + Z.imag ** 2
            if err < best_err:
                best_err = err
                best_C_load = c_l
                best_C_tune = c_t
    return np.array([best_C_load, best_C_tune])


def find_match(R_p, X_p, omega=OMEGA):
    """Numerically search (C_load, C_tune) so that Z_in = 50 + 0j.

    Coarse grid seed -> Newton refinement on real/imag residuals.
    """
    p = _seed_from_grid(R_p, X_p, omega=omega)
    for _ in range(60):
        C_load, C_tune = p
        Z = network_Zin(C_load, C_tune, R_p, X_p, omega=omega)
        r = np.array([Z.real - Z0, Z.imag])
        if np.linalg.norm(r) < 1e-6:
            break
        J = np.zeros((2, 2))
        for i in range(2):
            dp = np.zeros(2); dp[i] = max(1e-14, abs(p[i]) * 1e-4)
            Zp_ = network_Zin(*(p + dp), R_p=R_p, X_p=X_p, omega=omega)
            J[:, i] = (np.array([Zp_.real - Z0, Zp_.imag]) - r) / dp[i]
        try:
            step = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            break
        p_new = p - 0.7 * step
        p_new = np.clip(p_new, 1e-13, 5e-9)
        p = p_new
    return float(p[0]), float(p[1])


def quantize(value, step):
    return float(np.round(value / step) * step)


def make_dataset(n: int = 600):
    R_p_samples = 10.0 ** RNG.uniform(np.log10(2.0), np.log10(20.0), size=n)
    X_p_samples = RNG.uniform(-120.0, -10.0, size=n)
    C_load = np.zeros(n)
    C_tune = np.zeros(n)
    for i in range(n):
        c_l, c_t = find_match(R_p_samples[i], X_p_samples[i])
        C_load[i] = c_l
        C_tune[i] = c_t
    X = np.stack([C_load, C_tune], axis=-1).astype(np.float32)
    Y = np.stack([R_p_samples, X_p_samples], axis=-1).astype(np.float32)
    return X, Y


class InverseMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def fit_inverse(X, Y, epochs=200):
    # Normalize inputs and outputs.
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    ds = TensorDataset(torch.from_numpy(Xn.astype(np.float32)),
                       torch.from_numpy(Yn.astype(np.float32)))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    model = InverseMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return model, (X.mean(0), X.std(0) + 1e-9, Y.mean(0), Y.std(0) + 1e-9)


def apply_inverse(model_and_norm, X):
    model, (xm, xs, ym, ys) = model_and_norm
    Xn = (X - xm) / xs
    with torch.no_grad():
        Yn = model(torch.from_numpy(Xn.astype(np.float32))).numpy()
    return Yn * ys + ym


def main() -> None:
    print("generating (C_load, C_tune) <-> (R_p, X_p) pairs via numerical matching ...")
    X_train, Y_train = make_dataset(600)
    X_test, Y_test = make_dataset(120)
    model = fit_inverse(X_train, Y_train)

    pred = apply_inverse(model, X_test)
    err = pred - Y_test
    print(f"clean readouts:  R_p MAE = {np.mean(np.abs(err[:, 0])):.3f} ohm, "
          f"X_p MAE = {np.mean(np.abs(err[:, 1])):.3f} ohm")

    # Encoder quantisation noise (steps relevant for typical motorised VVCs).
    for step_pF in [1.0, 5.0, 20.0]:
        X_q = X_test.copy()
        for i in range(X_q.shape[0]):
            X_q[i, 0] = quantize(X_q[i, 0], step_pF * 1e-12)
            X_q[i, 1] = quantize(X_q[i, 1], step_pF * 1e-12)
        pred_q = apply_inverse(model, X_q)
        err_q = pred_q - Y_test
        print(f"  encoder step {step_pF:>4.1f} pF:  "
              f"R_p MAE = {np.mean(np.abs(err_q[:, 0])):.3f} ohm, "
              f"X_p MAE = {np.mean(np.abs(err_q[:, 1])):.3f} ohm")

    # One-off sanity check.
    R_p_true, X_p_true = 6.5, -45.0
    C_load, C_tune = find_match(R_p_true, X_p_true)
    Z_check = network_Zin(C_load, C_tune, R_p_true, X_p_true)
    pred_one = apply_inverse(model, np.array([[C_load, C_tune]]))[0]
    print(f"\nSanity: Z_p truth = {R_p_true:+.2f} {X_p_true:+.2f} j  =>  "
          f"C_load = {C_load*1e12:6.2f} pF, C_tune = {C_tune*1e12:6.2f} pF  =>  "
          f"|Z_in - 50| = {abs(Z_check - Z0):.4f} ohm, "
          f"inverse-model Z_p = {pred_one[0]:+.2f} {pred_one[1]:+.2f} j")


if __name__ == "__main__":
    main()
