# Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum Variable Capacitor Positions of Impedance Matching Unit

## Paper Info
- **Title**: Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum Variable Capacitor Positions of Impedance Matching Unit
- **Journal**: Electronics (MDPI), Vol. 14, Issue 10, 2022 (2025 issue)
- **Published**: May 2025
- **Link**: https://www.mdpi.com/2079-9292/14/10/2022

## Matching Keywords (>=2)
- Impedance Matching
- V-I Probe (Impedance Analyzer) — derived virtually from VVC positions instead of physical probe
- Plasma State Estimation (indirect sensor)
- Virtual Metrology

## Methods
The paper proposes a **virtual / sensor-less impedance monitor** for a 13.56 MHz capacitively coupled plasma (CCP) reactor that **derives the plasma load impedance directly from the positions of the two vacuum variable capacitors (VVCs)** of the matching network — without a physical V-I probe.

Pipeline:
1. **Calibration**: For a known 50 Ω terminated load, the matching network is exercised through its full (C_load, C_tune) range using the encoder-readouts of the stepper motors that drive the VVCs.
2. **Forward model**: An L-network model (or T/π) gives the input impedance Z_in of the matching circuit as a function of (C_load, C_tune, plasma load Z_p). When 50 Ω-matched, Z_in = 50 Ω, which gives one complex equation in Z_p.
3. **Inverse mapping**: For each (C_load, C_tune) at match, solve the L-network for Z_p analytically (closed-form) → R_p + jX_p. The mapping `(C_load, C_tune) → Z_p` is tabulated and validated against a reference Octiv VI probe.
4. **Real-time use**: During production runs, the VVC encoder positions are streamed at the matching network's tuning rate (~10 ms) and Z_p is reported as a virtual sensor.

## Conclusions
- The VVC-derived plasma impedance agrees with reference VI-probe measurements within a few percent across pressures of 5–100 mTorr and powers of 50–500 W.
- The technique enables real-time plasma impedance monitoring under 50 Ω-matched conditions **without an external sensor**, significantly reducing capex and integration complexity for fab fleets.
- The same readout (C_load, C_tune trajectory) works as a fingerprint for fault detection and process drift, offering virtual-metrology value beyond impedance inversion.

## Code Implementation
See `vvc_impedance_inverter.py` for an L-network forward model + closed-form inverse + a small calibration / verification routine.
