# Real-Time Equilibrium Reconstruction by Neural Network Based on HL-3 Tokamak (EFITNN)

- Keyword combination used for this folder: `Magnetic Equilibrium Reconstruction` + `Magnetic Flux Function` (+ `Equilibrium Fitting`)
- Routine date: 2026-05-20 (preprint posted 2024-05-18, within 2-year window 2024-05-20 .. 2026-05-20 — counted as eligible "recent" paper)

## Paper
- Title: Real-time equilibrium reconstruction by neural network based on HL-3 tokamak
- Authors: Z. Liu, X.-S. Wang, Y. Liu, K. Zhang, B. Li, B.-S. Yuan, X.-T. Yan, and others (HL-3 EFITNN team)
- arXiv: https://arxiv.org/abs/2405.11221
- Preprint year: 2024 (v1: 18 May 2024)

## Keyword match (>=2 required)
1. Magnetic Equilibrium Reconstruction — Direct replacement of the EFIT pipeline for HL-3 tokamak magnetic equilibrium.
2. Magnetic Flux Function — One of the network outputs is the high-resolution poloidal magnetic flux psi(R, Z), i.e. the magnetic flux function on the (R, Z) grid.
3. Equilibrium Fitting — The model is explicitly named "EFITNN" (Equilibrium FITting Neural Network) and is benchmarked against EFIT-output ground truth.

## Methods (summary)
- Dataset: 1159 HL-3 experimental discharges. Per time slice, inputs are 68 channels of magnetic measurements (poloidal-field probes, flux loops, plasma current Ip, loop voltage). Targets are the corresponding EFIT outputs: 8 global plasma parameters (Ip, beta_p, li, magnetic axis position, q95, etc.) + 2D poloidal flux psi(R, Z) + 1D toroidal current-density profile j_phi(rho).
- Architecture: a multi-task neural network with a shared trunk and three task-specific heads:
  - scalar head -> 8 global plasma parameters,
  - decoder head -> 2D poloidal flux map via successive transposed convolutions ("deconvolution"),
  - profile head -> 1D toroidal current density via 1D convolutions.
- Multi-task learning: the heads share the trunk and are trained jointly with weighted losses, which is reported to raise accuracy by up to 32% relative to independent single-task baselines (the heads "mutually correct" one another).
- Inference time: 0.08–0.45 ms per time slice on a single GPU, fast enough for in-shot isoflux feedback control.
- Validation: held-out discharges; comparison of psi(R, Z) and j_phi(rho) versus EFIT ground truth; tracking accuracy of global scalars across the H-mode operating space.

## Conclusions (as stated by the authors)
- EFITNN reproduces the magnetic equilibrium produced by EFIT — both global scalars and 2D flux maps — with errors small enough for real-time plasma-shape and profile control on HL-3.
- The multi-task structure (joint head for scalars, flux map, current profile) is the single biggest contributor to accuracy: enforcing consistency between Ip / li / beta_p and the 2D psi map regularizes each output.
- Inference at sub-millisecond latency makes the model suitable for inclusion in the HL-3 real-time control loop alongside isoflux control and disruption-prevention logic.
- The architectural pattern (sparse magnetic inputs → CNN/MLP trunk → multi-task heads producing flux map + profile + scalars) is portable to any tokamak with a similar magnetic-measurement layout.

## Code availability from the authors
The preprint does not link a public training/inference repository (HL-3 control-room data are not openly released). We therefore provide a self-contained reference implementation that mirrors the same EFITNN pattern on synthetic Grad–Shafranov-like flux maps.

## Reference code
See `reference_code.py` in this folder. It:
- generates synthetic poloidal flux maps psi(R, Z) on a 32x32 grid from a parametric (Ip, R0, a, kappa, delta) family inspired by Solov'ev-like equilibria;
- emulates 16 magnetic-probe + 4 flux-loop "measurements" by sampling psi and its gradients at fixed probe locations + adding noise;
- trains a small multi-task network that maps (probe signals, Ip, loop voltage) -> (scalar plasma parameters + 2D flux map + 1D current profile);
- reports per-task error on a held-out set and demonstrates the multi-task vs single-task gap.

Run: `python reference_code.py` (needs `numpy`, `torch`).
