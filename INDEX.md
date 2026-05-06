# Routine runs: Plasma Reconstruction / State Estimation

Each subfolder is named `keyword1_keyword2[_keyword3]` (the keyword combination used for the search). Each subfolder contains `report.md` (paper link, methods, conclusion) and `reference_code.py` (runnable reference implementation of the paper's core idea, since no public code accompanies the papers themselves).

Overall topic: Plasma Reconstruction and Mapping, Plasma State Estimation — Data-driven Plasma Modeling / Non-invasive Plasma Diagnostics based on Inverse Problems.

## Routine run 2026-04-23 (papers from 2024-04-23 .. 2026-04-23)

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `VirtualMetrology_PlasmaUniformityProfiling/` | Turica et al., "Reconstructions of electron-temperature profiles from EUROfusion Pedestal Database using turbulence models and machine learning", arXiv:2504.17486 | 2025 | Virtual Metrology, Plasma Uniformity Profiling, Spatially Resolved Mapping |
| `InverseProblemSolving_SurrogateModeling/` | Curvo, Ferreira, Jorge, "Using Deep Learning to Design High Aspect Ratio Fusion Devices", J. Plasma Phys. 91 E38 / arXiv:2409.00564 | 2024 | Inverse Problem Solving, Surrogate Modeling |
| `MultiPhysicsModeling_SurrogateModeling/` | Zhao et al., "Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR", arXiv:2510.16199 | 2025 | Multi-physics Modeling, Surrogate Modeling, Plasma Uniformity Profiling, Spatially Resolved Mapping |

## Routine run 2026-05-06 (papers from 2024-05-06 .. 2026-05-06)

New keyword combinations were chosen so as not to duplicate the previous run's combinations or papers.

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `MagneticEquilibriumReconstruction_EquilibriumFitting_MagneticFluxFunction/` | Cuizhi Zhou & Kaien Zhu, "Physics-Informed Neural Networks for High-Precision Grad-Shafranov Equilibrium Reconstruction", arXiv:2507.16636 | 2025 | Magnetic Equilibrium Reconstruction, Equilibrium Fitting, Magnetic Flux Function |
| `InverseProblemSolving_TomographicReconstruction/` | Öztürk, Akers, Pamela, MAST Team, Peers, Ghosh, "Inverse Rendering of Fusion Plasmas: Inferring Plasma Composition from Imaging Systems", Nuclear Fusion 65 026020 / arXiv:2408.07555 | 2024–25 | Inverse Problem Solving, Tomographic Reconstruction, Spatially Resolved Mapping |
| `OES_VirtualMetrology/` | Kang et al., "In-situ and Non-contact Etch Depth Prediction in Plasma Etching via Machine Learning (ANN & BNN) and Digital Image Colorimetry", arXiv:2505.03826 / Adv. Intell. Syst. 2025 | 2025 | OES, Virtual Metrology |

## Running the reference code
All reference scripts depend only on numpy + torch:

```
pip install numpy torch
# 2026-04-23 run
python VirtualMetrology_PlasmaUniformityProfiling/reference_code.py
python InverseProblemSolving_SurrogateModeling/reference_code.py
python MultiPhysicsModeling_SurrogateModeling/reference_code.py
# 2026-05-06 run
python MagneticEquilibriumReconstruction_EquilibriumFitting_MagneticFluxFunction/reference_code.py
python InverseProblemSolving_TomographicReconstruction/reference_code.py
python OES_VirtualMetrology/reference_code.py
```
