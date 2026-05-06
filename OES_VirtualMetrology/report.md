# In-situ and Non-contact Etch Depth Prediction in Plasma Etching via Machine Learning (ANN & BNN) and Digital Image Colorimetry

- Keyword combination used for this folder: `OES` + `Virtual Metrology` (also matches `Inverse Problem Solving` in the soft sense — predicting an unmeasured target from indirect signals)
- Routine date: 2026-05-06 (preprint 2025-05-03; journal Advanced Intelligent Systems 2025/2026; within 2-year window 2024-05-06 .. 2026-05-06)

## Paper
- Title: In-situ and Non-contact Etch Depth Prediction in Plasma Etching via Machine Learning (ANN & BNN) and Digital Image Colorimetry
- Authors: Minji Kang et al. (12+ co-authors)
- arXiv: https://arxiv.org/abs/2505.03826
- PDF: https://arxiv.org/pdf/2505.03826
- Journal: Advanced Intelligent Systems (Wiley), 2025, https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500517

## Keyword match (>=2 required)
1. OES (Optical Emission Spectroscopy) — In-situ OES is used in the paper to verify plasma stability across runs and is one of the indirect plasma sensors that motivates the data-driven virtual-metrology setup. Together with Digital Image Colorimetry (DIC) it provides the in-situ signal stream the ML models map to etch depth.
2. Virtual Metrology — The paper is a textbook virtual-metrology study: replace the costly post-process ellipsometry / SEM measurement of etch depth by a real-time prediction from process parameters and / or in-situ optical signals (OES + DIC). ANN and Bayesian Neural Network (BNN) regressors are trained as VM models.

## Methods (summary)
- Hardware: capacitively/inductively coupled SF₆ plasma etcher with ellipsometry mapping for the ground-truth etch depth, an in-situ OES probe for plasma stability checks, and a digital camera capturing RGB images of the wafer during/after processing for Digital Image Colorimetry (DIC).
- Dataset: a set of recipes scanning top RF power (50–110 W), chamber pressure (down to ~20 mTorr) and gas flow. For each recipe, etch depth is measured by ellipsometry as the regression target. Inputs are either (a) process parameters or (b) DIC colour features (R, G, B intensities) of the wafer image, or (c) both.
- Models:
  1. Artificial Neural Network (ANN) regressor: feed-forward, mapping process parameters → etch depth. Compared to a linear baseline, MSE drops significantly.
  2. Bayesian Neural Network (BNN) regressor: same input/output but with weight distributions, providing aleatoric + epistemic uncertainty.
  3. DIC-only regressor: uses the (R, G, B) triplet of the etched wafer as the *only* input and shows that colour alone — without process parameters — is enough to predict etch depth, confirming that the chemistry/plasma state is encoded in the optical appearance of the surface.
- Cross-checks: as top power increases from 50 → 110 W, the colour of the etched sample lightens monotonically, mirroring a higher etch rate; at 20 mTorr the colour change is most pronounced. This sanity-checks the colour ↔ etch-rate causality.

## Conclusions (as stated by the authors)
- The ANN trained on process parameters predicts etch depth with significantly lower MSE than a linear baseline; the BNN reproduces the same accuracy and additionally reports calibrated predictive uncertainty.
- DIC + ML alone is sufficient: the (R, G, B) intensities of the etched surface are a rich enough proxy to predict etch depth without explicit process parameters.
- The combination of in-situ OES (for plasma stability), DIC (for surface optical state) and ML regressors offers a low-cost, real-time, non-invasive virtual-metrology pipeline for plasma etching, contributing to better process stability and yield.

## Code availability from the authors
The arXiv preprint does not link a public code or data release. We therefore provide a self-contained PyTorch reference implementation that reproduces the *core* virtual-metrology setup: synthesise a small etch-depth dataset from a physically motivated mock recipe, train an ANN and a Bayesian-NN regressor, and report the MSE and uncertainty quality of the BNN.

## Reference code
See `reference_code.py` in this folder. It:
- generates a synthetic dataset of N≈300 plasma-etch runs with three process parameters (top RF power, pressure, flow) and produces ground-truth etch depths via a non-linear physical mock-up plus Gaussian process noise;
- additionally synthesises an RGB triplet per run via a smooth (etch-depth → colour) map (a thin-film interference-like sigmoid mixture) so the DIC-only experiment can be reproduced;
- trains:
  - a feed-forward ANN regressor (process params → etch depth),
  - a Monte-Carlo-dropout BNN approximation that yields predictive mean + standard deviation,
  - a DIC-only ANN that takes only (R, G, B) as input;
- reports test-set MSE and R² for all three models, and for the BNN also reports the empirical coverage of the 95 % predictive interval.

Run: `python reference_code.py` (CPU, < 30 s; PyTorch only).
