# Etch Rate Uniformity Monitoring for Photoresist Etch Using Multi-channel Optical Emission Spectroscopy and Scanning Floating Harmonic Probe in an Inductively Coupled Plasma Reactor

## Paper Info
- **Title**: Etch Rate Uniformity Monitoring for Photoresist Etch Using Multi-channel Optical Emission Spectroscopy and Scanning Floating Harmonic Probe in an Inductively Coupled Plasma Reactor
- **Journal**: Plasma Chemistry and Plasma Processing (Springer)
- **Published**: 2024 (DOI 10.1007/s11090-024-10498-0)
- **Link**: https://link.springer.com/article/10.1007/s11090-024-10498-0

## Matching Keywords (>=2)
- OES (Optical Emission Spectroscopy)
- Plasma Uniformity Profiling
- Spatially Resolved Mapping
- Plato/Langmuir Probe (Scanning Floating Harmonic Probe is a probe-class diagnostic)
- Virtual Metrology

## Methods
The authors instrument an industrial inductively coupled plasma (ICP) etcher with two synergistic spatially-resolved diagnostics:

1. **Multi-channel OES**: An array of fibre-coupled spectrometers views the chamber through several radial ports. The radial intensity of selected emission lines (e.g. Ar 750.4 nm for actinometry, O 777 nm and CFx bands as etch-product proxies) is fitted with a radial Abel-like inversion to recover a radial emissivity map.
2. **Scanning Floating Harmonic Probe (SFHP)**: A probe is moved across the wafer level recording the third-harmonic component of the floating-potential response to a sinusoidal RF excitation; this yields the local electron density `n_e(r)` and electron temperature `T_e(r)` without disturbing the discharge.

A **spatially-resolved etch-rate model** combines the OES-derived radical density (assumed proportional to the line ratio `I_O / I_Ar`) and the probe-derived ion flux `n_e * sqrt(T_e/m_i)` into a phenomenological etch-rate expression:
```
ER(r) = a * Γ_radical(r) + b * Γ_ion(r) ^ γ + c
```
The coefficients `(a, b, γ, c)` are fitted by regression against post-etch wafer-thickness profiles measured by a contact profilometer / ellipsometer over a design-of-experiment sweep of pressure / power / gas mixture.

## Conclusions
- The fitted spatially-resolved model predicts the photoresist etch rate with **R² = 0.99 and MAPE = 1.3 %** on held-out conditions, and predicts the etch-rate uniformity (radial standard deviation) with **R² = 0.99 and MAPE = 12.0 %**.
- Combining OES-only or probe-only diagnostics yields significantly worse predictions than the combined model, confirming that ion and radical fluxes are independent uniformity drivers.
- The diagnostic stack functions as an in-situ virtual metrology that replaces destructive wafer-level measurements during process development.

## Code Implementation
See `oes_fhp_uniformity_model.py` for a synthetic data generator + scikit-learn regression pipeline that mirrors the etch-rate uniformity model structure.
