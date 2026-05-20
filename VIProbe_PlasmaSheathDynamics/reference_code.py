"""
Reference implementation inspired by:
  J. W. Brooks, R. Dutta, "Sheath thickness measurements with the biased
  plasma impedance probe: Agreement with Child-Langmuir scaling",
  arXiv:2602.08743, 2026.

We replicate the *inverse-fit* component of the paper:
  1. Forward model:  Z(omega; n_e, nu_en, s) for a probe + sheath + plasma
     lumped-element circuit, with the bulk plasma described by a cold-plasma
     dielectric.
  2. Synthetic broadband Z(omega) sweeps with additive noise.
  3. Inversion: fit (n_e, nu_en, s) to each sweep via a hand-rolled
     Levenberg-Marquardt solver (numpy only -- no scipy dependency).
  4. Scan the DC bias V_b, compare the recovered sheath thickness s(V_b)
     to the Child-Langmuir prediction, and report the empirical scaling.

Run: python reference_code.py     (needs numpy)
"""

from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31
KB_EV = 1.0                        # we'll treat Te in eV directly

RNG = np.random.default_rng(11)

OMEGAS = 2.0 * np.pi * np.linspace(1e6, 5e9, 200)   # 1 MHz .. 5 GHz


def cold_plasma_eps(omega, n_e, nu_en):
    omega_p2 = n_e * QE ** 2 / (EPS0 * ME)
    return 1.0 - omega_p2 / (omega * (omega + 1j * nu_en))


def forward_Z(omega, n_e, nu_en, s, T_e=2.0, area=1e-4, L_p=5e-9):
    """Probe impedance with a vacuum-like sheath layer in series with the
    bulk plasma admittance through a fictitious geometric capacitance."""
    eps_p = cold_plasma_eps(omega, n_e, nu_en)
    C_geom = EPS0 * area / 1e-3                           # mm-scale gap
    Y_bulk = 1j * omega * eps_p * C_geom                  # plasma admittance
    C_sheath = EPS0 * area / max(s, 1e-7)
    Z_sheath = 1.0 / (1j * omega * C_sheath)
    Z_probe = 1j * omega * L_p
    return Z_probe + Z_sheath + 1.0 / Y_bulk


def child_langmuir_thickness(V_b, n_e, T_e=2.0):
    """Standard CL sheath thickness for ions: s_CL ~ lambda_D * (eV_b/T_e)^(3/4)."""
    lam_D = np.sqrt(EPS0 * T_e / (n_e * QE))
    return (2.0 / 3.0) ** 0.5 * lam_D * (np.maximum(V_b, 0.0) / T_e) ** 0.75


def residuals(params, omega, Z_meas):
    log_n, log_nu, log_s = params
    n_e = 10.0 ** log_n
    nu_en = 10.0 ** log_nu
    s = 10.0 ** log_s
    Z_hat = forward_Z(omega, n_e, nu_en, s)
    r = np.concatenate([(Z_meas.real - Z_hat.real), (Z_meas.imag - Z_hat.imag)])
    return r / np.max(np.abs(r) + 1e-12)


def jacobian(params, omega, Z_meas, eps: float = 1e-3):
    r0 = residuals(params, omega, Z_meas)
    J = np.zeros((r0.size, len(params)))
    for i in range(len(params)):
        dp = np.zeros_like(params); dp[i] = eps
        J[:, i] = (residuals(params + dp, omega, Z_meas) - r0) / eps
    return J, r0


def levmar(p0, omega, Z_meas, max_iter: int = 60, lam: float = 1e-2):
    p = np.array(p0, dtype=float)
    for _ in range(max_iter):
        J, r = jacobian(p, omega, Z_meas)
        H = J.T @ J + lam * np.eye(len(p))
        g = J.T @ r
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        p_new = p + step
        r_new = residuals(p_new, omega, Z_meas)
        if np.linalg.norm(r_new) < np.linalg.norm(r):
            p = p_new
            lam *= 0.7
        else:
            lam *= 2.5
        if np.linalg.norm(step) < 1e-6:
            break
    return p


def main() -> None:
    n_e_true = 1.0e16
    nu_en_true = 5.0e7
    T_e = 2.0

    biases = np.linspace(5.0, 60.0, 8)
    recovered = []
    truth_with_correction = []
    cl_pred = []

    for V_b in biases:
        s_phys = 0.74 * child_langmuir_thickness(V_b, n_e_true, T_e=T_e)   # synthetic alpha=0.74
        Z = forward_Z(OMEGAS, n_e_true, nu_en_true, s_phys)
        noise = (RNG.standard_normal(Z.shape) + 1j * RNG.standard_normal(Z.shape)) \
                * 0.005 * np.abs(Z).mean()
        Z_meas = Z + noise

        # Multi-start LM: seeding sheath thickness near the CL prediction is
        # what the experimental procedure also does (use bias + estimated n_e
        # to seed the search). Keep the best fit.
        best = None
        for s_seed in np.linspace(0.4, 1.0, 5) * child_langmuir_thickness(V_b, n_e_true, T_e=T_e):
            p_init = [np.log10(n_e_true), np.log10(nu_en_true), np.log10(s_seed)]
            p_fit = levmar(p_init, OMEGAS, Z_meas)
            r_norm = np.linalg.norm(residuals(p_fit, OMEGAS, Z_meas))
            if best is None or r_norm < best[1]:
                best = (p_fit, r_norm)
        s_fit = 10.0 ** best[0][2]
        recovered.append(s_fit)
        truth_with_correction.append(s_phys)
        cl_pred.append(child_langmuir_thickness(V_b, n_e_true, T_e=T_e))

    recovered = np.array(recovered)
    cl_pred = np.array(cl_pred)
    alpha = float((recovered / cl_pred).mean())
    rms = float(np.sqrt(np.mean(((recovered - 0.74 * cl_pred) / (0.74 * cl_pred)) ** 2)))

    print("V_b [V]   s_truth [mm]   s_fit [mm]   s_CL [mm]")
    for v, s_t, s_f, s_cl in zip(biases, truth_with_correction, recovered, cl_pred):
        print(f"{v:6.1f}    {s_t*1e3:8.3f}     {s_f*1e3:8.3f}    {s_cl*1e3:8.3f}")
    print(f"\nempirical alpha = <s_fit / s_CL> = {alpha:.3f}  (target ~ 0.74)")
    print(f"relative RMS error vs 0.74 * CL  = {rms*100:.1f} %")


if __name__ == "__main__":
    main()
