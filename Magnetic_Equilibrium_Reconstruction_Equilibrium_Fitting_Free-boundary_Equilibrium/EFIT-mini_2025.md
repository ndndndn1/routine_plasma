# EFIT-mini: An Embedded, Multi-task Neural Network-driven Equilibrium Inversion Algorithm

## Paper Info
- **Title**: EFIT-mini: An Embedded, Multi-task Neural Network-driven Equilibrium Inversion Algorithm
- **Tokamak**: EXL-50U Spherical Torus
- **Published**: March 2025 (arXiv preprint)
- **arXiv ID**: 2503.19467
- **Link**: https://arxiv.org/abs/2503.19467
- **Companion work (PTEFIT)**: https://arxiv.org/html/2601.12378

## Matching Keywords (>=2)
- Magnetic Equilibrium Reconstruction
- Equilibrium Fitting
- Free-boundary Equilibrium
- Magnetic Flux Function
- Inverse Problem Solving

## Methods
EFIT-mini is a real-time inversion algorithm that combines an **embedded multi-task neural network** with a **single Picard iteration** of the free-boundary Grad-Shafranov solver to reconstruct the magnetic equilibrium of a tokamak.

Pipeline:
1. **Inputs**: 68 channels of magnetic measurements (poloidal field probes, flux loops, plasma current Ip, loop voltage Vloop, PF coil currents) on EXL-50U.
2. **Multi-task NN**: outputs key parameters that parameterize the source `J_phi(R,Z; alpha,beta)` of the Grad-Shafranov (GS) PDE (e.g. plasma current profile coefficients, p'(psi), FF'(psi) basis weights, plasma boundary descriptors).
3. **Single Picard step**: solves `Δ*ψ = -μ0 R J_phi` with the predicted source on a 129x129 (R,Z) grid using a precomputed Green's-function inverse, producing a self-consistent flux function ψ(R,Z) and last-closed-flux-surface (LCFS).
4. **Multi-task loss**: weighted MSE on profile parameters + ψ-map L2 + LCFS Chamfer distance + plasma boundary geometric descriptors (R_axis, Z_axis, kappa, delta).

Key design points:
- Integration of inductive bias from the GS PDE means the NN does not have to learn the full nonlinear `Ψ` field from scratch; it learns only the parameter manifold.
- Embedded deployment: model exported to TensorRT/ONNX for the EXL-50U real-time control PC.

## Conclusions
- **Accuracy**: > 98% LCFS overlap ratio vs. offline EFIT on EXL-50U validation discharges.
- **Speed**: 0.36 ms per time slice at 129×129 resolution (≈ 2.7 kHz throughput).
- **PTEFIT extension**: production deployment running at 1 kHz during EXL-50U discharges, capturing the X-divertor minus → X-divertor plus transition immediately after start-up — a regime the offline reconstruction misses because of its latency.
- The hybrid (NN + 1 Picard) approach generalises better than pure-NN ψ-image regressors at the limit of training-distribution edge cases such as near-LCFS topology changes.

## Code Implementation
A simplified PyTorch reference implementation that captures the multi-task NN + single-Picard step idea is provided in `efit_mini_reference.py`.
