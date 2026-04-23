# Routine run 2026-04-23: Plasma Reconstruction / State Estimation

Trigger: papers published within 2 years of routine-run date (2024-04-23 .. 2026-04-23).
Overall topic: Plasma Reconstruction and Mapping, Plasma State Estimation — Data-driven Plasma Modeling / Non-invasive Plasma Diagnostics based on Inverse Problems.

Each subfolder is named `keyword1_keyword2[_keyword3]` (the keyword combination used for the search). Each subfolder contains `report.md` (paper link, methods, conclusion) and `reference_code.py` (runnable reference implementation of the paper's core idea, since no public code accompanied the papers themselves).

## Papers included (new for this routine, no overlap with prior runs)

| Folder | Paper | Year | Primary keywords matched |
|---|---|---|---|
| `VirtualMetrology_PlasmaUniformityProfiling/` | Turica et al., "Reconstructions of electron-temperature profiles from EUROfusion Pedestal Database using turbulence models and machine learning", arXiv:2504.17486 | 2025 | Virtual Metrology, Plasma Uniformity Profiling, Spatially Resolved Mapping |
| `InverseProblemSolving_SurrogateModeling/` | Curvo, Ferreira, Jorge, "Using Deep Learning to Design High Aspect Ratio Fusion Devices", J. Plasma Phys. 91 E38 / arXiv:2409.00564 | 2024 | Inverse Problem Solving, Surrogate Modeling, (Invertible Neural Networks in spirit via MDN) |
| `MultiPhysicsModeling_SurrogateModeling/` | Zhao et al., "Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR", arXiv:2510.16199 | 2025 | Multi-physics Modeling, Surrogate Modeling, Plasma Uniformity Profiling, Spatially Resolved Mapping |

## Running the reference code
All three scripts depend only on numpy + torch:

```
pip install numpy torch
python VirtualMetrology_PlasmaUniformityProfiling/reference_code.py
python InverseProblemSolving_SurrogateModeling/reference_code.py
python MultiPhysicsModeling_SurrogateModeling/reference_code.py
```
