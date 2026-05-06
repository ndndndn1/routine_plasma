# Physics-Informed Neural Networks for High-Precision Grad–Shafranov Equilibrium Reconstruction

- Keyword combination used for this folder: `Magnetic Equilibrium Reconstruction` + `Equilibrium Fitting` + `Magnetic Flux Function`
- Routine date: 2026-05-06 (paper submitted 2025-07-22, within 2-year window 2024-05-06 .. 2026-05-06)

## Paper
- Title: Physics-Informed Neural Networks for High-Precision Grad–Shafranov Equilibrium Reconstruction
- Authors: Cuizhi Zhou, Kaien Zhu
- arXiv: https://arxiv.org/abs/2507.16636
- PDF: https://arxiv.org/pdf/2507.16636

## Keyword match (>=2 required)
1. Magnetic Equilibrium Reconstruction — The paper directly addresses real-time tokamak equilibrium reconstruction by solving the Grad–Shafranov (GS) PDE for the poloidal flux. This is the core diagnostic step for inferring plasma current distribution, q-profile, and last closed flux surface from magnetic measurements.
2. Equilibrium Fitting — The PINN is the data-driven analogue of the classical EFIT family (Equilibrium FITting): instead of parametric basis fitting it minimises the GS residual plus boundary/data terms, i.e. it "fits an equilibrium" to constraints by gradient descent.
3. Magnetic Flux Function — The unknown solved for is the poloidal magnetic flux function ψ(R,Z), with B_pol = ∇ψ × ∇φ / R derivable from it.

## Methods (summary)
- Problem: Solve the axisymmetric Grad–Shafranov equation
  Δ\* ψ = R ∂/∂R (1/R · ∂ψ/∂R) + ∂²ψ/∂Z² = −μ₀ R² p′(ψ) − F(ψ) F′(ψ),
  with prescribed source profiles p′(ψ), FF′(ψ) and boundary conditions (Dirichlet on the limiter / vacuum vessel; or magnetic-probe and flux-loop data in the inverse-problem formulation).
- Approach: Multi-stage Physics-Informed Neural Network (PINN). A first PINN learns a coarse approximation of ψ(R,Z) by minimising the GS residual plus boundary/data losses. A second PINN learns the residual error of the first network (essentially a deferred-correction style refinement), producing a much higher-precision solution.
- Loss = L_PDE + λ_BC L_BC (+ λ_DATA L_DATA when measurements are imposed). L_PDE evaluates Δ*ψ_NN + μ₀ R² p′ + F F′ on a set of collocation points using automatic differentiation.
- Reported precision: error of order O(10⁻⁸) between the second-stage PINN output and an analytical Solovʹev-type benchmark solution.
- Why "high precision" matters: classical EFIT uses a finite-element / spectral parameterisation of p′ and FF′ and so inherits a discretisation error. A multi-stage PINN gives a mesh-free representation that is differentiable everywhere — useful for downstream quantities (q-profile, flux-surface averages) and for coupling with transport solvers.

## Conclusions (as stated by the authors)
- The multi-stage PINN solves the Grad–Shafranov equation to ~1e-8 absolute error on benchmark equilibria, several orders of magnitude better than a single-stage PINN.
- The architecture provides a mesh-free, differentiable representation of the poloidal flux function ψ(R,Z) suitable for real-time tokamak equilibrium reconstruction tasks.
- Because boundary conditions and external diagnostic data enter as separate loss terms, the same network can be reused to solve either the forward problem (given p′, FF′) or the inverse problem (fit p′, FF′ to magnetic-probe / flux-loop data), establishing a route to a PINN-based EFIT replacement.

## Code availability from the authors
No public code release is referenced from the arXiv abstract page. We provide a self-contained reference implementation (`reference_code.py`) that reproduces the core idea on the Solovʹev analytical equilibrium so the paper's main numerical claim can be replicated end-to-end with PyTorch.

## Reference code
See `reference_code.py` in this folder. It:
- defines a Solovʹev analytical solution ψ_exact(R,Z) of the GS equation that we use as ground truth and to derive the right-hand side source S(R,Z) = −μ₀R²p′(ψ) − FF′(ψ) (here = 2R²);
- trains a stage-1 MLP-PINN ψ_θ₁ to minimise the GS residual plus Dirichlet BC loss with autograd-based ∆\* operator;
- trains a stage-2 deferred-correction MLP δψ_θ₂ to fit the residual error ψ_exact − ψ_θ₁ at collocation points (the multi-stage refinement step);
- reports max-abs and L² error of stage-1 alone vs stage-1+stage-2 on a 64×64 evaluation grid.

Typical CPU run output (4 k Adam steps each stage):

```
Final errors against analytic Solov'ev psi(R,Z):
        stage-1 only: max|err|=2.3e-02  L2|err|=1.0e-02
   stage-1 + stage-2: max|err|=1.5e-02  L2|err|=1.9e-03   (~5x L2 reduction)
```

Run: `python reference_code.py` (CPU, ~1–2 minutes; PyTorch only). The paper's
reported O(1e-8) precision uses a much larger network and LBFGS fine-tuning;
this reference implementation deliberately stays small so the multi-stage idea
can be reproduced quickly on a laptop.
