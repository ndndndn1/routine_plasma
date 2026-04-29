# Tomography for Plasma Imaging: A Unifying Framework for Bayesian Inference

## Paper Info
- **Title**: Tomography for Plasma Imaging: a Unifying Framework for Bayesian Inference
- **Published**: June 2025 (arXiv preprint)
- **arXiv ID**: 2506.20232
- **Link**: https://arxiv.org/abs/2506.20232 (HTML: https://arxiv.org/html/2506.20232v1)

## Matching Keywords (>=2)
- Tomographic Reconstruction
- Inverse Problem Solving
- Spatially Resolved Mapping
- Plasma Uniformity Profiling (radial emissivity profile)

## Methods
This paper presents a unifying perspective on **sparse-view tomographic reconstructions of plasma emissivity** — a strongly ill-posed linear inverse problem encountered in fusion bolometers, soft-X-ray (SXR) cameras, and Hα cameras.

Forward model: line integrals through the plasma cross-section
```
b_i = ∫_{L_i} ε(r,z) dl   →   b = K ε + n,    n ~ N(0, Σ_n)
```
where `K` is the sparse geometric matrix and `ε(r,z)` is the local emissivity to recover.

Bayesian framework:
- **Likelihood** `p(b | ε) ∝ exp(-1/2 ||b − Kε||^2_{Σ_n^{-1}})`.
- **Prior** `p(ε) ∝ exp(-Φ(ε))` encodes spatial smoothness, anisotropy along magnetic flux surfaces, non-negativity, and sparsity.
- **MAP estimate** = solution to a regularised least-squares; equivalent to Tikhonov / Phillips-Tikhonov / Minimum Fisher Information / total-variation when Φ is chosen as Sobolev / Fisher-information / TV norms.
- **Posterior sampling**: pCN MCMC and Laplace approximation give posterior covariance, hence pixel-wise uncertainty maps.
- **Neural priors**: deep image prior and score-based diffusion priors are placed in the same Φ slot, recovering recent learning-based reconstructions as MAP under a learned prior.

The paper compares Tikhonov, Bayesian (Gaussian Process / Gauss-Markov field), and learned (CNN / score-based) reconstructions on standardised JET and TCV datasets, and shows that all of them are special cases of a single hierarchical Bayesian objective.

## Conclusions
- A single Bayesian framework subsumes Tikhonov regularisation, Phillips-Tikhonov, Minimum Fisher Information, anisotropic smoothing along flux surfaces, neural-network reconstructors, and diffusion-prior reconstructions.
- Treating the prior as a tunable component allows transparent comparison and combination of model-based and data-driven reconstructions.
- The framework gives free uncertainty quantification (posterior covariance), which is critical for downstream control and disruption avoidance.

## Code Implementation
See `bayesian_plasma_tomography.py` for a self-contained 2D fan-beam reconstruction with three priors (Tikhonov L2, total variation, anisotropic Gauss-Markov along flux surfaces) and Laplace-posterior uncertainty.
