# Sheath Thickness Measurements with the Biased Plasma Impedance Probe: Agreement with Child–Langmuir Scaling

- Keyword combination used for this folder: `V-I Probe (Impedance Analyzer)` + `Plasma Sheath Dynamics`
- Routine date: 2026-05-20 (preprint posted 2026-02-09, within 2-year window)

## Paper
- Title: Sheath thickness measurements with the biased plasma impedance probe: Agreement with Child–Langmuir scaling
- Authors: J. W. Brooks, R. Dutta
- arXiv: https://arxiv.org/abs/2602.08743
- Preprint date: 9 Feb 2026

## Keyword match (>=2 required)
1. V-I Probe (Impedance Analyzer) — The probe is an RF plasma impedance probe (PIP) that measures the complex impedance Z(omega) of an antenna immersed in the plasma; broadband V–I response is used to extract plasma parameters.
2. Plasma Sheath Dynamics — The target quantity is the DC sheath thickness around the biased probe, which depends on the ion-saturation current and bulk plasma density (Child–Langmuir scaling).

## Methods (summary)
- Hardware: a plasma impedance probe is driven with a swept-frequency RF signal (broadband) and an additional DC bias voltage is applied to the same electrode. The impedance Z(omega) is recorded across a wide band so that bulk-plasma and sheath signatures appear at different frequency regions.
- Inverse model: the broadband Z(omega) curve is fitted with a single analytical expression that simultaneously yields plasma density n_e, electron-neutral damping (electron collisional drag), and sheath thickness s. This is a single-step inverse fit rather than two stacked single-frequency inversions, which reduces sensitivity to noise.
- Bias scan: at each operating point, the DC bias is scanned while the broadband impedance is recorded. The resulting sheath-thickness vs bias curve is compared against the Child–Langmuir (CL) prediction s_CL(V_b, n_e, T_e).
- Cross-validation: the bulk plasma density obtained from the same broadband fit (independent of bias) confirms that biasing the probe does not significantly perturb the bulk plasma, supporting the validity of the biased-PIP approach.
- Experimental conditions: multiple discharge conditions (varying gas pressure, RF power) are tested to span a range of sheath thicknesses.

## Conclusions (as stated by the authors)
- Across the explored conditions, the biased-PIP sheath thickness follows CL scaling with a single empirical correction factor alpha ≈ 0.74 (i.e. s_meas ≈ 0.74 · s_CL). The constancy of alpha across conditions is the key experimental finding.
- Biasing the probe does not significantly perturb the bulk plasma density, validating PIP as a non-invasive sheath diagnostic.
- The biased PIP is positioned as a complementary diagnostic to the Langmuir probe: PIP gives sheath thickness directly with minimal model assumptions, whereas Langmuir analysis must infer thickness from an I–V trace via multi-step modeling.
- The result tightens experimental support for classical CL sheath models and provides a recipe for real-time sheath monitoring in low-temperature plasmas (e.g. semiconductor process chambers, electric-propulsion thrusters).

## Code availability from the authors
The preprint is an experimental study and does not include public code. We provide a reference Python implementation of the *inverse model*: a forward model for the broadband Z(omega) of a probe-plus-sheath circuit and a non-linear least-squares routine that recovers (n_e, nu_en, sheath thickness) from a noisy synthetic Z(omega) trace.

## Reference code
See `reference_code.py` in this folder. It:
- builds a lumped-element forward model Z(omega) = R_sheath(s) + (j omega L_p − 1/(j omega C_sheath(s))) + plasma admittance with a cold-plasma dielectric epsilon_p(omega; n_e, nu_en);
- emulates a broadband sweep + bias scan with synthetic noise;
- fits each Z(omega) trace for (n_e, nu_en, s) using SciPy-style Levenberg–Marquardt (implemented by hand on top of numpy);
- compares the recovered s(V_b) against the Child–Langmuir prediction and reports the empirical scaling factor alpha.

Run: `python reference_code.py` (needs `numpy`).
