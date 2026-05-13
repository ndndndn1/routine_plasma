# Neural network reconstruction of the DIII-D tokamak plasma boundary using a reduced set of diagnostics

- **Authors:** M. Stokolesov et al. (DIII-D team)
- **Venue / Date:** arXiv:2505.10709 (May 2025); subsequently published in *Journal of Plasma Physics* (Cambridge).
- **Link:** https://arxiv.org/abs/2505.10709
- **Routine date:** 2026-05-13 (within 2-year window: 2024-05-13 .. 2026-05-13)

## Keyword matches (>=2 required)
1. **Free-boundary Equilibrium** — the task is to reconstruct the *last closed flux surface* (LCFS), i.e., the free boundary of the plasma, without solving the inverse Grad–Shafranov problem with magnetic probes.
2. **Spatially Resolved Mapping** — the network outputs the LCFS as a contour of points around the poloidal cross-section, i.e., a spatial map of the plasma shape.
3. **Inverse Problem Solving** — the paper explicitly frames LCFS reconstruction from coil currents (with optional Ip and loop voltage) as an *ill-posed inverse problem*.
4. **Plasma State Estimation** — relevant because the LCFS is the central state variable for shape control.

## Methods

**Motivation.** Future Fusion Pilot Plants (FPPs) place tritium-breeding blankets and neutron shielding between the plasma and most magnetic diagnostics. Many of the magnetic probes and flux loops EFIT relies on will not be available. The paper asks: with only *external* signals (PF coil currents, optionally plasma current and loop voltage), can a neural network still recover the plasma boundary?

**Inputs / outputs.**
- *Input model A:* the vector of PF coil currents only (≈20 channels).
- *Input model B:* coil currents + Ip + Vloop.
- *Output:* the LCFS represented as the (R, Z) coordinates of a fixed number of contour points around the boundary (the boundary is sampled at a fixed set of angles around the geometric axis).

**Architecture.** Plain Multi-Layer Perceptron (MLP). The authors deliberately use a simple feedforward network rather than a sequence or graph model because the input is a low-dimensional, well-aligned vector and the output is a fixed-length shape descriptor.

**Training data.** Roughly **25,000 DIII-D discharges (2004–2024)**, after filtering for missing EFIT signals, durations < 1500 ms, and time-step gaps > 100 ms. The filtered dataset contains ~5 M time slices. Both positive- and negative-triangularity discharges are included to test generalization across plasma shapes. Targets are provided by traditional EFIT reconstructions.

**Loss.** Mean squared error on (R, Z) of the LCFS contour points (equivalently, mean point-displacement).

**Held-out evaluation.** Mean point displacement on a discharge-disjoint test set:
- Model A (coils only): **0.04 m**.
- Model B (coils + Ip + Vloop): **0.03 m**.

## Conclusions

- LCFS reconstruction is achievable from external diagnostics alone, demonstrating a viable shape-sensing path for FPP-class devices that cannot host the full EFIT diagnostic suite.
- Adding two cheap, externally observable signals (Ip, Vloop) yields a ~25 % reduction in mean boundary error.
- The work is part of a broader DIII-D / GA programme on *reconstruction-free* magnetic control: instead of reconstructing the full ψ map and feeding it to a controller, the boundary itself can be predicted directly and used as the controller's state estimate.
- Limitations: the model is trained on EFIT labels, so it inherits EFIT's biases; extrapolation outside the historical operating envelope is not guaranteed.

## Code

No public repository is referenced in the paper. `reference_code.py` in this folder is an original, self-contained PyTorch implementation that:
1. Builds a synthetic DIII-D-shaped tokamak in normalized units with random elongation, triangularity, and Shafranov-shifted axis.
2. Simulates a small set of PF coil currents via a least-squares projection of the boundary geometry (Biot–Savart-style stand-in).
3. Trains an MLP to recover the LCFS as a 64-point contour from (coils) vs. (coils + Ip + Vloop), matching the paper's two-model comparison.
