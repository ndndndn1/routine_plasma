# Physics Insights from a Large-Scale 2D UEDGE Simulation Database for Detachment Control in KSTAR

- Keyword combination used for this folder: `Multi-physics Modeling` + `Surrogate Modeling` (+ `Plasma Uniformity Profiling`, `Spatially Resolved Mapping`)
- Routine date: 2026-04-23 (paper published 2025-10-17, within 2-year window)

## Paper
- Title: Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR
- Authors: Menglong Zhao, Xueqiao Xu, Ben Zhu, Thomas Rognlien, Xinxing Ma, William Meyer, KyuBeen Kwon, David Eldon, and collaborators
- Preprint (arXiv): https://arxiv.org/abs/2510.16199
- Hugging Face Papers: https://hf.co/papers/2510.16199
- Dataset record (DOE Data Explorer, KSTAR_UEDGE_Dataset): https://www.osti.gov/dataexplorer/biblio/dataset/2997129

## Keyword match (>=2 required)
1. Multi-physics Modeling — UEDGE is a 2D multi-fluid edge-plasma + neutral-particle + impurity radiation code with magnetic and electric drifts; the database couples plasma transport, atomic physics, and neutral dynamics across the scrape-off layer and divertor.
2. Surrogate Modeling — The paper motivates and feeds the DivControlNN neural surrogate (trained on the same database) for real-time detachment control; the dataset is explicitly built for surrogate-model training.
3. Plasma Uniformity Profiling — Strike-point Te, upstream density, and impurity radiation front position are mapped as functions of actuation.
4. Spatially Resolved Mapping — Each of the ~70,000 entries is a 2D field solution (ne, Te, Ti, impurity radiation, neutral density) on the UEDGE mesh.

## Methods (summary)
- Simulation code: UEDGE 2D multi-fluid edge/divertor model on the KSTAR geometry (with and without the new W divertor upgrade). Physics included: cross-field anomalous transport, parallel transport, neutral recycling, impurity radiation (Ne/Ar seeding), ExB and diamagnetic drifts.
- Scan dimensions (5 knobs): upstream separatrix density n_u, input power P_in, plasma current I_p, impurity fraction c_imp, anomalous cross-field transport coefficients (chi, D).
- Solutions: ~70,000 converged steady states; subset of time-dependent runs with gas-puff ramps to characterise the transient response.
- Physics diagnostics extracted from each run: strike-point Te_sp, peak heat flux q_pk, divertor radiation fraction, in-out asymmetry, position of the low-field-side (LFS) and high-field-side (HFS) radiation fronts.
- Time-dependent study: delays of 5-15 ms at the outer strike point and ~40 ms for the LFS radiation front in response to gas puffing.
- Surrogate (DivControlNN): neural regressor (dataset card on OSTI) trained on the 5 control knobs -> detachment indicators, intended for real-time model-based control.

## Conclusions (as stated by the authors)
- A robust detachment indicator emerges: strike-point electron temperature Te_sp ~ 3-4 eV at detachment onset, largely independent of upstream conditions across the scanned range. This is a simple, spatially resolved signature that a surrogate or diagnostic reconstruction can target.
- KSTAR shows distinctive in-out divertor asymmetries that differ qualitatively from DIII-D, attributed to magnetic-topology and drift effects in the scanned configurations.
- Transient time-scales provide the control horizon: ~10 ms for strike-point quantities, ~40 ms for the LFS radiation front, setting the bandwidth requirements on any detachment controller.
- The database is explicitly designed as the training substrate for DivControlNN, a neural surrogate that provides ms-level inference for real-time detachment control in KSTAR — a concrete example of surrogate modeling enabling plasma-state estimation from indirect (upstream) sensors.
- Implication for reconstruction/mapping: pairing a spatially resolved 2D multi-physics database with a conditional neural surrogate turns slow physics simulators into fast observers that can invert actuator/sensor readings into plasma-state estimates.

## Code / data availability
- Dataset: KSTAR_UEDGE_Dataset on DOE Data Explorer (see link above). Intended for training the DivControlNN surrogate model.
- UEDGE code itself is open-source via LLNL (https://github.com/LLNL/UEDGE).
- A trained DivControlNN is referenced in the authors' companion works; to keep this reproducible without the large dataset, we ship a compact surrogate demo below.

## Reference code
See `reference_code.py`. It:
- builds a synthetic detachment dataset that mimics UEDGE inputs (n_u, P_in, I_p, c_imp, chi) and outputs (Te_sp, q_pk, f_rad, LFS front position) with qualitative scalings consistent with the paper (Te_sp saturates near 3-4 eV at detachment onset);
- trains a small MLP surrogate that maps 5 actuators -> 4 detachment indicators;
- reports held-out MAE and plots Te_sp vs n_u with the 3-4 eV detachment band highlighted.

Run: `python reference_code.py` (PyTorch required).
