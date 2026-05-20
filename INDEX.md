# Routine runs: Plasma Reconstruction / State Estimation

Overall topic: Plasma Reconstruction and Mapping, Plasma State Estimation — Data-driven Plasma Modeling / Non-invasive Plasma Diagnostics based on Inverse Problems.

Each subfolder is named `keyword1_keyword2[_keyword3]` (the keyword combination used for the search). Each subfolder contains `report.md` (paper link, methods, conclusion) and `reference_code.py` (runnable reference implementation of the paper's core idea, since no public code accompanied the papers themselves).

## Routine run 2026-04-23 (papers within 2024-04-23 .. 2026-04-23)

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `VirtualMetrology_PlasmaUniformityProfiling/` | Turica et al., "Reconstructions of electron-temperature profiles from EUROfusion Pedestal Database using turbulence models and machine learning", arXiv:2504.17486 | 2025 | Virtual Metrology, Plasma Uniformity Profiling, Spatially Resolved Mapping |
| `InverseProblemSolving_SurrogateModeling/` | Curvo, Ferreira, Jorge, "Using Deep Learning to Design High Aspect Ratio Fusion Devices", J. Plasma Phys. 91 E38 / arXiv:2409.00564 | 2024 | Inverse Problem Solving, Surrogate Modeling, (Invertible Neural Networks in spirit via MDN) |
| `MultiPhysicsModeling_SurrogateModeling/` | Zhao et al., "Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR", arXiv:2510.16199 | 2025 | Multi-physics Modeling, Surrogate Modeling, Plasma Uniformity Profiling, Spatially Resolved Mapping |

## Routine run 2026-05-20 (papers within 2024-05-20 .. 2026-05-20, no overlap with prior run)

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `OES_InverseProblemSolving/` | Wang & Zhu, "Development of optical emission spectroscopy method with neural network model: Case study of determining the electron density in a xenon microwave discharge", J. Appl. Phys. 136, 243302 | 2024 | OES, Inverse Problem Solving, Surrogate Modeling |
| `MagneticEquilibriumReconstruction_MagneticFluxFunction/` | Z. Liu et al., "Real-time equilibrium reconstruction by neural network based on HL-3 tokamak" (EFITNN), arXiv:2405.11221 | 2024 | Magnetic Equilibrium Reconstruction, Magnetic Flux Function, Equilibrium Fitting |
| `VIProbe_PlasmaSheathDynamics/` | Brooks & Dutta, "Sheath thickness measurements with the biased plasma impedance probe: Agreement with Child–Langmuir scaling", arXiv:2602.08743 | 2026 | V-I Probe (Impedance Analyzer), Plasma Sheath Dynamics |
| `ImpedanceMatching_VirtualMetrology/` | "Impedance Monitoring of Capacitively Coupled Plasma Based on the Vacuum Variable Capacitor Positions of Impedance Matching Unit", MDPI Electronics 14(10), 2022 | 2025 | Impedance Matching, Virtual Metrology, (MKS Instruments-style hardware) |
| `MicrowaveInterferometry_InverseProblemSolving/` | "A microwave reflection diagnostics of inhomogeneous plasma distribution based on multi-peak points phenomenon", Phys. Plasmas 32, 042113 | 2025 | Microwave Interferometry, Inverse Problem Solving, Spatially Resolved Mapping, Plasma Sheath Dynamics |

## Running the reference code

All scripts depend on `numpy` (+ `torch` for the ML-heavy ones):

```
pip install numpy torch

# routine 2026-04-23
python VirtualMetrology_PlasmaUniformityProfiling/reference_code.py
python InverseProblemSolving_SurrogateModeling/reference_code.py
python MultiPhysicsModeling_SurrogateModeling/reference_code.py

# routine 2026-05-20
python OES_InverseProblemSolving/reference_code.py
python MagneticEquilibriumReconstruction_MagneticFluxFunction/reference_code.py
python VIProbe_PlasmaSheathDynamics/reference_code.py
python ImpedanceMatching_VirtualMetrology/reference_code.py
python MicrowaveInterferometry_InverseProblemSolving/reference_code.py
```
