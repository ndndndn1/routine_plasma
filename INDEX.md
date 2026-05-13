# Plasma Reconstruction / State Estimation — Routine Index

Overall topic: Plasma Reconstruction and Mapping, Plasma State Estimation — Data-driven Plasma Modeling / Non-invasive Plasma Diagnostics based on Inverse Problems.

Each subfolder is named `keyword1_keyword2[_keyword3]` (the keyword combination used for the search). Each subfolder contains `report.md` (paper link, methods, conclusion) and `reference_code.py` (runnable reference implementation of the paper's core idea, since no public code accompanied the papers themselves).

---

## Routine run 2026-05-13 (this run)

Trigger: papers published within 2 years of routine-run date (2024-05-13 .. 2026-05-13).
Deduped against prior routine (2026-04-23). All three picks are 2025 magnetic-equilibrium reconstruction papers covering the inverse-problem / shape-mapping spectrum.

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `MagneticEquilibriumReconstruction_EquilibriumFitting/` | Zheng et al., "EFIT-mini: An Embedded, Multi-task Neural Network-driven Equilibrium Inversion Algorithm", arXiv:2503.19467 | 2025 | Magnetic Equilibrium Reconstruction, Equilibrium Fitting, Inverse Problem Solving, Magnetic Flux Function |
| `FreeBoundaryEquilibrium_SpatiallyResolvedMapping/` | Stokolesov et al., "Neural network reconstruction of the DIII-D tokamak plasma boundary using a reduced set of diagnostics", arXiv:2505.10709 | 2025 | Free-boundary Equilibrium, Spatially Resolved Mapping, Inverse Problem Solving |
| `MagneticFluxFunction_InverseProblemSolving/` | Zhou & Zhu, "Physics-Informed Neural Networks for High-Precision Grad-Shafranov Equilibrium Reconstruction", arXiv:2507.16636 | 2025 | Magnetic Flux Function, Inverse Problem Solving, Equilibrium Fitting, Magnetic Equilibrium Reconstruction |

## Routine run 2026-04-23 (previous run)

Trigger: 2024-04-23 .. 2026-04-23.

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `VirtualMetrology_PlasmaUniformityProfiling/` | Turica et al., "Reconstructions of electron-temperature profiles from EUROfusion Pedestal Database using turbulence models and machine learning", arXiv:2504.17486 | 2025 | Virtual Metrology, Plasma Uniformity Profiling, Spatially Resolved Mapping |
| `InverseProblemSolving_SurrogateModeling/` | Curvo, Ferreira, Jorge, "Using Deep Learning to Design High Aspect Ratio Fusion Devices", J. Plasma Phys. 91 E38 / arXiv:2409.00564 | 2024 | Inverse Problem Solving, Surrogate Modeling, (Invertible Neural Networks in spirit via MDN) |
| `MultiPhysicsModeling_SurrogateModeling/` | Zhao et al., "Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR", arXiv:2510.16199 | 2025 | Multi-physics Modeling, Surrogate Modeling, Plasma Uniformity Profiling, Spatially Resolved Mapping |

## Running the reference code

All scripts depend only on numpy + torch (CPU is enough).

```
pip install numpy torch

# 2026-05-13 run
python MagneticEquilibriumReconstruction_EquilibriumFitting/reference_code.py
python FreeBoundaryEquilibrium_SpatiallyResolvedMapping/reference_code.py
python MagneticFluxFunction_InverseProblemSolving/reference_code.py

# 2026-04-23 run
python VirtualMetrology_PlasmaUniformityProfiling/reference_code.py
python InverseProblemSolving_SurrogateModeling/reference_code.py
python MultiPhysicsModeling_SurrogateModeling/reference_code.py
```
