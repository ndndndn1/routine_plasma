# Reconstructions of Electron-Temperature Profiles from EUROfusion Pedestal Database using Turbulence Models and Machine Learning

- Keyword combination used for this folder: `Virtual Metrology` + `Plasma Uniformity Profiling` (+ `Spatially Resolved Mapping`)
- Routine date: 2026-04-23 (paper published 2025-04-24, within 2-year window)

## Paper
- Title: Reconstructions of electron-temperature profiles from EUROfusion Pedestal Database using turbulence models and machine learning
- Authors: L.-P. Turica, A. R. Field, L. Frassinetti, A. A. Schekochihin, JET Contributors, EUROfusion Tokamak Exploitation Team
- Venue: Journal of Plasma Physics (Cambridge Core, 2025)
- Preprint (arXiv): https://arxiv.org/abs/2504.17486
- Hugging Face Papers: https://hf.co/papers/2504.17486
- Journal page: https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/reconstructions-of-electrontemperature-profiles-from-eurofusion-pedestal-database-using-turbulence-models-and-machine-learning/1F6EB5B07C60212FB7D2923927CD24F1

## Keyword match (>=2 required)
1. Virtual Metrology — The electron-temperature profile is not directly measured but reconstructed ("virtually" predicted) from electron-density profile and engineering set-points (Bt, Ip, gas fuelling rate, strike-point configuration, etc.). This is the classic virtual-metrology setup: indirect sensors in, hard-to-measure target out.
2. Plasma Uniformity Profiling — The target is the radial pedestal profile of Te including pedestal top, width and position, i.e. uniformity along the edge transport barrier.
3. Spatially Resolved Mapping — The ML model outputs a spatially resolved Te(psi_N) profile at the pedestal, not just a scalar.

## Methods (summary)
- Dataset: EUROfusion JET ITER-Like-Wall (ILW) pedestal database of H-mode ELMy pulses. Each record contains fitted pedestal profiles (mtanh) for electron density ne(psi_N) and electron temperature Te(psi_N), plus engineering parameters (Bt, Ip, P_NBI, gas rate, triangularity, plasma volume, strike-point config, etc.).
- Input to ML: ne(psi_N) profile (or its mtanh parameters: n_ped, width, position, separatrix value) plus engineering scalars. 80% of the database used for training, 20% held out.
- Target: Te(psi_N) profile (or mtanh parameters for Te: T_ped, width, position).
- Models: a feed-forward regression is benchmarked against physics-based heat-flux scalings (gyrokinetic ETG-motivated Q_e(nabla T, T, n, B, ...)). The ML model takes more inputs than the analytic scalings and outperforms them when only ne and engineering parameters are available.
- Figures of merit: prediction of Te profile within ~20% of the experimental value across the test set; correct pedestal width and position; ranking of engineering parameters by feature importance.
- Physics cross-check: derived pedestal heat-flux scalings from nonlinear gyrokinetic simulations (tau = Zeff Te/Ti scan) are fitted to JET data; best-fit coefficients are reported. These physics models need additional experimental inputs to match ML accuracy.

## Conclusions (as stated by the authors)
- A simple ML regressor, given engineering parameters and the ne profile, reconstructs JET-ILW pedestal Te profiles to within ~20% for the held-out 20% of pulses, reproducing pedestal width and location.
- Most important engineering inputs for Te pedestal reconstruction are: magnetic-field strength, particle fuelling rate, plasma current, and strike-point configuration.
- Theory-based ETG heat-flux scalings fit the data qualitatively but require extra experimental inputs (e.g. heat flux and main-ion temperature) to reach ML accuracy — so pure data-driven surrogates are complementary to, not yet replaced by, first-principles models.
- Implication for diagnostics: with density profiles (e.g. from interferometry / Thomson) and control-room engineering values, the Te pedestal profile can be inferred non-invasively, which is useful for plasma-state estimation and for providing a fast "virtual sensor" for pedestal Te on shots where Te diagnostics are missing or noisy.

## Code availability from the authors
No official code release is linked in the abstract/paper page (publication in JPP). We therefore provide a self-contained reference implementation that reproduces the setup: pedestal mtanh parametrization, synthetic ne/Te training pairs, and a small MLP regressor that maps (ne_profile + engineering params) -> Te_profile with uncertainty.

## Reference code
See `reference_code.py` in this folder. It:
- generates a synthetic but physically plausible pedestal database (mtanh profiles for ne, Te with engineering-parameter dependence);
- builds an MLP that maps (ne profile + engineering scalars) -> Te profile;
- trains on 80%, evaluates on 20%;
- reports mean relative error and plots a held-out Te reconstruction vs ground truth.

Run: `python reference_code.py` (PyTorch required).
