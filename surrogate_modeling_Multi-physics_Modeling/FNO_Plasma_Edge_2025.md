# Neural Operator Surrogate Models of Plasma Edge Simulations: Feasibility and Data Efficiency

## Paper Info
- **Title**: Neural operator surrogate models of plasma edge simulations: feasibility and data efficiency
- **Published**: February 2025 (arXiv preprint, v2)
- **arXiv ID**: 2502.17386
- **Link**: https://arxiv.org/abs/2502.17386 (HTML: https://arxiv.org/html/2502.17386v2)

## Matching Keywords (>=2)
- surrogate modeling
- Multi-physics Modeling (JOREK MHD + STORM turbulence)
- Plasma Sheath Dynamics (edge / scrape-off layer)
- Magnetic Equilibrium Reconstruction (input prior to STORM)

## Methods
The authors evaluate **Fourier Neural Operators (FNOs)** as fast surrogate models for two leading edge-plasma codes:

1. **JOREK** (3D non-linear MHD for tokamak ELMs and disruptions)
2. **STORM** (2D fluid turbulence in the scrape-off layer)

The FNO is trained on **single-step rollouts**: given the state at time `t`, predict the state at `t + Δt`. At inference time the network is rolled out auto-regressively to produce trajectories.

Key contributions:
- **Data-efficiency study**: how many high-fidelity simulations are needed to reach a given long-rollout error level. Answer: low hundreds of trajectories suffice, but accuracy degrades with rollout length unless temporal bundling or noise injection is used.
- **Transfer learning**: pretrain on cheap **low-resolution / low-fidelity** simulations and fine-tune on a few **high-fidelity** ones. This yields an **order-of-magnitude reduction** in error for small high-fidelity datasets.
- **Conditioning**: equilibrium fields (B, q, ψ from Grad-Shafranov solvers) and boundary conditions (sheath / wall fluxes) are concatenated as additional channels.

## Conclusions
- FNO surrogates can reproduce JOREK and STORM trajectories at **two to three orders of magnitude lower wallclock cost** than the underlying solvers, with acceptable accuracy on bulk integrated quantities.
- Long-term auto-regressive rollouts remain challenging: a hybrid of FNO + physics correction or temporal bundling is needed for stable multi-millisecond predictions.
- Transfer learning from low-fidelity to high-fidelity is the most effective lever for reducing high-fidelity data requirements — important because each JOREK / STORM run can cost 10⁴–10⁵ CPU-h.

## Code Implementation
See `fno_plasma_edge.py` for a minimal Fourier Neural Operator surrogate that ingests multi-channel 2D fields (n, T, ψ) and predicts the next time step, with a transfer-learning routine.
