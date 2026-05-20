# Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum Variable Capacitor Positions of the Impedance Matching Unit

- Keyword combination used for this folder: `Impedance Matching` + `Virtual Metrology`
- Routine date: 2026-05-20 (paper published 2025-05, within 2-year window)

## Paper
- Title: Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum Variable Capacitor Positions of Impedance Matching Unit
- Venue: MDPI Electronics, Vol. 14, Issue 10, Article 2022 (May 2025)
- Article landing page: https://www.mdpi.com/2079-9292/14/10/2022
- ResearchGate mirror: https://www.researchgate.net/publication/391794095_Impedance_Monitoring_of_Capacitively_Coupled_Plasma_Based_on_the_Vacuum_Variable_Capacitor_Positions_of_Impedance_Matching_Unit
- DOI prefix: 10.3390/electronics14102022

## Keyword match (>=2 required)
1. Impedance Matching — The diagnostic exploits the very thing whose state is normally hidden inside the chamber: the matching unit itself. The position of the two vacuum variable capacitors (VVCs) of the L-type matching network is treated as the measurement vector.
2. Virtual Metrology — Bulk plasma impedance Z_p = R_p + jX_p is *not* measured by a dedicated VI probe; instead it is inferred from the readily available control-side signal (VVC positions) under the 50 Ω-matched condition. This is the canonical virtual-metrology premise: replace an in-chamber sensor with a model that reads control-room signals.
3. (Bonus) MKS Instruments — VVC-based matching networks are the standard matching topology used in industrial plasma chambers, including units from MKS Instruments and ENI/AMC.

## Methods (summary)
- Setup: a single-frequency (13.56 MHz) capacitively coupled plasma chamber driven through an L-type matching network with two motorized vacuum variable capacitors C_load and C_tune. Under 50 Ω match the RF source sees a real 50 Ω load, but the chamber-side impedance Z_p depends on plasma conditions (power, pressure, gas chemistry, wafer state).
- Forward circuit model: starting from the matching-network topology and the assumed 50 Ω condition, the two VVC capacitances (C_load, C_tune) can be inverted analytically into the complex load impedance Z_p seen at the matchbox output. The mapping is one-to-one because the L-type network has exactly two degrees of freedom.
- Data: VVC positions are read out continuously from the matching-unit encoders during plasma operation across a process recipe. The inferred Z_p is compared against:
  - a reference VI probe (when available) inserted upstream of the matchbox,
  - the expected qualitative behavior of CCP impedance with pressure / power / sheath state.
- Sensitivity / noise study: the analytical inversion is differentiated to express dZ_p / d(C_load, C_tune), giving the measurement noise floor as a function of VVC encoder resolution. The required encoder resolution to achieve a target dZ_p uncertainty is reported.
- Process correlation: the inferred Z_p is shown to track sheath/bulk-state changes that should accompany changes in power, pressure, and gas mixture, demonstrating its use as a process fingerprint.

## Conclusions (as stated by the authors)
- Under the 50 Ω-matched condition, the two VVC positions of the L-type matching network are sufficient to reconstruct the complex plasma impedance Z_p without any external VI probe — i.e. the matching unit itself becomes the diagnostic.
- The reconstructed Z_p is in good agreement with reference VI-probe measurements and shows the expected sensitivities to RF power, pressure, and gas chemistry, so it can be used as a real-time, non-invasive plasma-state indicator.
- This is attractive for production tools where adding an in-chamber sensor is intrusive: it converts an existing control-side signal into a virtual sensor, enabling continuous plasma-state monitoring with no extra hardware.

## Code availability from the authors
None linked. We provide a reference implementation of the analytical inversion: a small forward model of the L-type matching network and the closed-form inversion (C_load, C_tune) -> Z_p, together with a sensitivity calculation and a noise-floor estimate.

## Reference code
See `reference_code.py` in this folder. It:
- defines the L-network forward model Z_in(C_load, C_tune; Z_p, omega, L_s) and solves for the (C_load, C_tune) that achieve a 50 Ω match;
- exposes a closed-form inverse that recovers Z_p from observed (C_load, C_tune);
- evaluates the inversion on a synthetic process recipe (sweeping plasma R_p and X_p);
- emulates encoder quantization on the VVC readouts and reports the resulting Z_p uncertainty (a virtual-metrology noise floor).

Run: `python reference_code.py` (needs `numpy`).
