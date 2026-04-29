"""
Etch-rate uniformity surrogate built from multi-channel OES line ratios and a
scanning floating harmonic probe (SFHP). The ER(r) model is

    ER(r) = a * Gamma_rad(r) + b * Gamma_ion(r)^gamma + c

where Gamma_rad(r) is approximated from I_O / I_Ar actinometry and
Gamma_ion(r) = n_e(r) * sqrt(k T_e(r) / m_i).

Run: python oes_fhp_uniformity_model.py
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_percentage_error

K_B = 1.380649e-23
M_I = 6.63e-26          # ~ Ar+ ion mass [kg], close enough for O2/Ar etch


def synthetic_diagnostic(n_runs=120, n_radii=15, seed=0):
    rng = np.random.default_rng(seed)
    r = np.linspace(0, 0.15, n_radii)                # 0..150 mm
    runs = []
    for _ in range(n_runs):
        p = rng.uniform(5, 30)        # mTorr
        P = rng.uniform(300, 1200)    # W
        f_O2 = rng.uniform(0.0, 0.3)  # O2 fraction in Ar/O2
        # Toy radial profiles: ne and Te peaked off-axis, Te flatter
        ne = 1e16 * (1 + 0.3 * np.cos(np.pi * r / 0.18)) * (P / 800)
        Te = 3.0 * (1 + 0.05 * np.cos(np.pi * r / 0.18))
        I_O = 5e3 * f_O2 * (1 + 0.2 * (r / 0.15))
        I_Ar = 1e4 * (1 + 0.05 * (r / 0.15))
        Gamma_rad = I_O / I_Ar                       # actinometric proxy
        Gamma_ion = ne * np.sqrt(K_B * Te * 11600 / M_I)
        # Ground-truth etch rate
        a, b, gamma, c = 5e3, 2e-19, 0.85, 5.0
        er = a * Gamma_rad + b * Gamma_ion ** gamma + c
        er += rng.normal(0, 1.0, size=er.shape)
        runs.append((p, P, f_O2, r, ne, Te, Gamma_rad, Gamma_ion, er))
    return runs


def fit_model(runs):
    Grad = np.concatenate([rn[6] for rn in runs])
    Gion = np.concatenate([rn[7] for rn in runs])
    ER = np.concatenate([rn[8] for rn in runs])

    def model(X, a, b, gamma, c):
        gr, gi = X
        return a * gr + b * gi ** gamma + c

    popt, _ = curve_fit(model, (Grad, Gion), ER, p0=(1.0, 1e-19, 1.0, 0.0), maxfev=10000)
    pred = model((Grad, Gion), *popt)
    return popt, pred, ER


def uniformity(profile):
    return profile.std() / profile.mean()


if __name__ == "__main__":
    runs = synthetic_diagnostic(n_runs=200)
    popt, pred, ER = fit_model(runs)
    print("Fitted (a, b, gamma, c):", popt)
    print("Pointwise R^2 :", r2_score(ER, pred))
    print("Pointwise MAPE:", mean_absolute_percentage_error(ER, pred))

    # Uniformity prediction (per-run radial std/mean)
    u_true, u_pred = [], []
    idx = 0
    for rn in runs:
        n = len(rn[3])
        u_true.append(uniformity(rn[8]))
        u_pred.append(uniformity(pred[idx:idx + n]))
        idx += n
    u_true = np.array(u_true)
    u_pred = np.array(u_pred)
    print("Uniformity R^2 :", r2_score(u_true, u_pred))
    print("Uniformity MAPE:", mean_absolute_percentage_error(u_true, u_pred))
