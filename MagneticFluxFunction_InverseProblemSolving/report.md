# Physics-Informed Neural Networks for High-Precision Grad–Shafranov Equilibrium Reconstruction

- **Authors:** Cuizhi Zhou, Kaien Zhu (Nankai University)
- **Venue / Date:** arXiv:2507.16636, submitted 22 Jul 2025
- **Link:** https://arxiv.org/abs/2507.16636
- **Routine date:** 2026-05-13 (within 2-year window: 2024-05-13 .. 2026-05-13)

## Keyword matches (>=2 required)
1. **Magnetic Flux Function** — the network directly approximates the poloidal flux function ψ(R, Z) that solves the Grad–Shafranov equation.
2. **Inverse Problem Solving** — the GS equation with prescribed source profiles p′(ψ), ff′(ψ) is the *direct* problem; the multi-stage scheme is built to converge with extreme precision on the inverse of the Grad–Shafranov differential operator.
3. **Equilibrium Fitting** — the result is a high-precision equilibrium fit to a known analytical solution (Solov'ev), positioning the method as a successor to traditional fixed-boundary equilibrium codes.
4. **Magnetic Equilibrium Reconstruction** — the abstract explicitly frames the work as "a core step in real-time diagnostic tasks in fusion research."

## Methods

**Problem.** Solve the Grad–Shafranov equation
R ∂/∂R ((1/R) ∂ψ/∂R) + ∂²ψ/∂Z² = −μ₀ R² p′(ψ) − ff′(ψ)
on a poloidal cross-section, with Dirichlet boundary conditions on a chosen domain. The benchmark target is the Solov'ev analytical family of equilibria, which provides closed-form ψ for verification.

**Vanilla data-free PINN.** A neural network ψ_θ(R, Z) is trained to minimize the residual of the GS equation on collocation points in the interior plus a boundary-fitting term — there is no labelled data, only the PDE itself. This is the approach of Kaltsas (arXiv:2311.13491, Phys. Plasmas 2024). Errors hover around O(10⁻⁵).

**Key contribution: multi-stage PINN.** The authors observe that vanilla PINNs plateau at relative errors of about 10⁻⁵ because the residual loss landscape becomes ill-conditioned once the bulk solution is captured. They split training into two stages:

1. **Stage 1.** Train ψ_θ₁(R, Z) with the standard PINN objective until convergence. Error to the analytical Solov'ev solution: ~O(10⁻⁵).

2. **Stage 2 (correction network).** Freeze ψ_θ₁. Define a second network ψ_θ₂(R, Z) that learns the *residual* between ψ_θ₁ and the true ψ. Equivalently, the second network targets the small high-frequency error left by stage 1. Loss is the GS residual evaluated on (ψ_θ₁ + ψ_θ₂), which is now dominated by O(10⁻⁵) terms — easier to fit. Error after stage 2: ~O(10⁻⁸).

This recipe is conceptually a *Picard-style* refinement using a second neural network as the corrector. Three orders of magnitude improvement is achieved without any change to the architecture or optimizer.

**Architecture.** Standard fully-connected MLP, tanh activations, sinusoidal positional features on (R, Z). Both stages share this template.

**Training setup.** Collocation points are sampled on a rectangular (R, Z) domain. Loss = MSE of GS residual on interior collocation points + MSE of boundary mismatch on boundary points. Adam → L-BFGS hybrid optimization. Verified on Solov'ev profiles with closed-form ψ.

## Conclusions

- A two-stage PINN scheme drives the equilibrium reconstruction error to ~10⁻⁸, three orders of magnitude better than single-stage PINNs.
- The improvement comes from decoupling "coarse" and "fine" components of ψ across two networks, sidestepping the conditioning problem of single-stage residual minimization.
- The method is *data-free*: no EFIT labels required. It produces a continuous, differentiable ψ(R, Z) that can be queried at arbitrary points — useful for downstream tasks (field-line tracing, transport solvers) that need higher resolution than a 129×129 grid.
- Limitation: tested only on fixed-boundary Solov'ev analytics; the free-boundary, diagnostic-constrained case (as in EFIT-mini) is not addressed.

## Code

No public code repository is referenced in the paper. `reference_code.py` in this folder is an original, self-contained PyTorch implementation that reproduces the two-stage PINN idea on a Solov'ev test case (psi = (R² − 1)²/8 + Z²/2, source = R² + 1). On a CPU in ~3 minutes it demonstrates:
- Stage 1 (Adam, 3 000 steps): mean |ψ_pred − ψ_true| ≈ 1.0 × 10⁻³.
- Stage 2 (frozen Stage 1 + correction net + L-BFGS polish): ≈ 7 × 10⁻⁴, a 1.4–1.5× reduction.

The paper reports a much larger reduction (~10⁻⁵ → ~10⁻⁸) that requires much longer training and tighter L-BFGS schedules; the reference script is meant to illustrate the *staging trick* rather than match those numbers in a 3-minute run.
