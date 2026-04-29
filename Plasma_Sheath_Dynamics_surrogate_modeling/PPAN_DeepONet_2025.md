# Deep Transfer Operator Learning for Predicting Low-Temperature Plasma Sheath Dynamics in Semiconductor Processing

## Paper Info
- **Title**: Deep transfer operator learning for predicting low temperature plasma sheath dynamics in semiconductor processing
- **Authors**: Ahn, Bae, Yoo, Nam
- **Journal**: Physics of Plasmas, Vol. 32, 093505 (2025)
- **Published**: September 2025
- **Link**: https://pubs.aip.org/aip/pop/article/32/9/093505/3361615/

## Matching Keywords (>=2)
- Plasma Sheath Dynamics
- surrogate modeling
- Inverse Problem Solving (predicts sheath state given pressure / power / gas)
- Multi-physics Modeling

## Methods
The authors propose a **Plasma Physics-Aware Network (PPAN)** built on the **DeepONet** operator-learning architecture with **transfer learning** for predicting sheath-region plasma parameters in inductively coupled plasma (ICP) reactors used in semiconductor processing.

Key elements:
1. **Source domain**: high-fidelity 1D fluid simulations of capacitively coupled discharges (CCP) covering broad parameter sweeps. The **branch** network of the DeepONet ingests reactor controls (pressure p, RF power P_RF, frequency f, gas mixture). The **trunk** network ingests the spatial coordinate `x` (or `(x,t)` for time-resolved variant).
2. **Targets**: spatial profiles of sheath-region quantities — sheath voltage V_s, sheath thickness s, ion flux Γ_i, ion energy distribution function (IEDF) at the wafer surface.
3. **Physics-aware regularisation**: the loss enforces ambipolarity, Bohm-criterion at the sheath edge, and total-current continuity through dedicated penalty terms — making the DeepONet honour first-principle constraints near the sheath edge where pure data fits typically degrade.
4. **Transfer learning**: pretrained DeepONet on the CCP source domain is fine-tuned with **only 5–15 ICP samples** to adapt to the target inductively coupled reactor — orders of magnitude fewer simulations than training from scratch.
5. **Domain split**: a sheath/bulk segmentation gate routes the operator to two specialised heads, sharply resolving the steep gradients in the sheath without smearing them across the quasi-neutral bulk.

## Conclusions
- PPAN matches the reference 1D fluid simulator within a few percent on sheath voltage, sheath thickness and IEDF moments.
- Transfer learning yields high accuracy on ICP with as few as 5–15 fine-tuning samples; without pretraining ~100× more data is required.
- The physics-informed sheath/bulk segmentation is essential — pure DeepONets without it under-resolve the sheath edge and violate the Bohm criterion.
- The surrogate is fast enough (sub-millisecond per query) to serve as an in-the-loop replacement for the fluid solver inside semiconductor virtual-metrology and process-control workflows.

## Code Implementation
See `ppan_deeponet_reference.py` for a minimal physics-aware DeepONet skeleton with sheath/bulk gating and a pretraining-then-fine-tune routine.
