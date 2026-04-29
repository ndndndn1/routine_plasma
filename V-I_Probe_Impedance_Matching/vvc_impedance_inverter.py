"""
Sensor-less plasma impedance monitor: derive Z_plasma directly from
matching-network VVC positions (C_load, C_tune) under 50-ohm match.

Topology assumed (typical L-network for CCP, 13.56 MHz):

    50 Ohm -- L_series -- C_tune ----+---- C_load -- (plasma load Z_p)
                                     |
                                    GND

Forward model: input impedance seen from the generator must equal 50 Ohm
when matched. Inverting that complex equation gives Z_p in closed form.

Run: python vvc_impedance_inverter.py
"""

import numpy as np

OMEGA = 2 * np.pi * 13.56e6
L_SERIES = 1.0e-6        # Henry, generator-side series inductor
Z0 = 50.0                # ohm


def Z_input(C_tune, C_load, Zp, omega=OMEGA, L=L_SERIES):
    """Input impedance of the L-network terminated by Zp."""
    Y_load = 1j * omega * C_load + 1.0 / Zp
    Z_after_Cload = 1.0 / Y_load
    Z_after_Ctune = Z_after_Cload + 1.0 / (1j * omega * C_tune)
    Z_in = 1j * omega * L + Z_after_Ctune
    return Z_in


def invert_match(C_tune, C_load, omega=OMEGA, L=L_SERIES, Z_target=Z0 + 0j):
    """Solve Z_input = Z_target for Zp."""
    Z_residual = Z_target - 1j * omega * L - 1.0 / (1j * omega * C_tune)
    Y_after_Cload = 1.0 / Z_residual
    Y_p = Y_after_Cload - 1j * omega * C_load
    Zp = 1.0 / Y_p
    return Zp


def calibrate_lookup_table(c_tune_range, c_load_range):
    """Build a (C_tune, C_load) -> Z_plasma table assuming match."""
    table = []
    for ct in c_tune_range:
        for cl in c_load_range:
            zp = invert_match(ct, cl)
            table.append((ct, cl, zp.real, zp.imag))
    return np.array(table)


if __name__ == "__main__":
    Zp_truth = 5 - 80j        # [ohm], typical CCP impedance
    # Find (C_tune, C_load) that matches to 50 ohm by simple grid search
    cts = np.linspace(20e-12, 1000e-12, 500)
    cls = np.linspace(20e-12, 1000e-12, 500)
    best = None
    best_err = np.inf
    for ct in cts:
        for cl in cls:
            err = abs(Z_input(ct, cl, Zp_truth) - Z0)
            if err < best_err:
                best_err, best = err, (ct, cl)
    ct_m, cl_m = best
    Zp_est = invert_match(ct_m, cl_m)
    print(f"VVC at match: C_tune={ct_m * 1e12:.1f} pF, C_load={cl_m * 1e12:.1f} pF")
    print(f"True   Zp = {Zp_truth}")
    print(f"Inverted Zp = {Zp_est:.3f}")
    print(f"|residual on 50-ohm match| = {best_err:.3e}")
