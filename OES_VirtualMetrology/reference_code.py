"""Reference implementation of the virtual-metrology pipeline from
Kang et al., "In-situ and Non-contact Etch Depth Prediction in Plasma Etching
via Machine Learning (ANN & BNN) and Digital Image Colorimetry"
(arXiv:2505.03826 / Advanced Intelligent Systems 2025).

We reproduce the central setup on a synthetic dataset:
  * generate N etch-process runs with three process parameters (top RF power,
    pressure, gas flow) and a noisy non-linear ground-truth etch depth;
  * derive a Digital-Image-Colorimetry-style RGB triplet from etch depth via a
    smooth thin-film-interference-inspired mapping;
  * train and evaluate
       (a) a feed-forward ANN  (process params -> etch depth),
       (b) a Monte-Carlo-dropout BNN approximation (process params -> mean+std),
       (c) an RGB-only ANN  (DIC features -> etch depth);
  * report test MSE / R^2 / 95% interval coverage.

This demonstrates the "OES + Virtual Metrology" idea (indirect optical sensors
mapping to a hard-to-measure target) without depending on any proprietary
fab data.

Run:
    pip install numpy torch
    python reference_code.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Synthetic etch dataset
# ---------------------------------------------------------------------------
#   Process parameters (normalised to [0,1]):
#     p[0] : top RF power     (50 W ... 110 W)
#     p[1] : pressure         (5 mTorr ... 50 mTorr)  -- inverted (low p = high)
#     p[2] : SF6 flow rate    (10 sccm ... 50 sccm)
#   Etch depth (nm) follows a saturating non-linear law:
#     d(p) = D0 * (1 - exp(-k * power)) * sqrt(flow) / (1 + alpha*pressure)
# ---------------------------------------------------------------------------
def make_etch_dataset(n: int = 320, noise: float = 0.04):
    p = torch.rand(n, 3, device=DEVICE)  # [0,1] normalised process params
    power = 50.0 + 60.0 * p[:, 0]        # W
    pressure = 5.0 + 45.0 * p[:, 1]      # mTorr
    flow = 10.0 + 40.0 * p[:, 2]         # sccm
    D0 = 220.0                            # max etch depth (nm)
    k = 0.04                              # power saturation coefficient
    alpha = 0.02                          # pressure suppression
    d_clean = D0 * (1.0 - torch.exp(-k * power)) * torch.sqrt(flow / 50.0) / (
        1.0 + alpha * pressure
    )
    d = d_clean * (1.0 + noise * torch.randn(n, device=DEVICE))
    return p, d, d_clean


# ---------------------------------------------------------------------------
# Mock Digital Image Colorimetry (DIC) RGB feature from etch depth.
# Thin-film interference produces a roughly periodic colour signature with
# depth; we approximate it by a smooth sigmoid-mixture over (R,G,B) channels.
# ---------------------------------------------------------------------------
def etch_to_rgb(d: torch.Tensor) -> torch.Tensor:
    # Three Gaussian-ish "colour bands" centred at different depths
    centres = torch.tensor([60.0, 130.0, 200.0], device=d.device)
    widths = torch.tensor([35.0, 35.0, 35.0], device=d.device)
    diff = (d.unsqueeze(-1) - centres) / widths
    rgb = torch.exp(-0.5 * diff**2)  # (n, 3) in [0,1]
    # add small camera noise
    rgb = rgb + 0.01 * torch.randn_like(rgb)
    return rgb.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ANN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = hidden
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_regressor(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                    steps: int = 1500, lr: float = 1e-3) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        pred = model(X)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def eval_metrics(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    pred = model(X)
    mse = F.mse_loss(pred, y).item()
    ss_res = ((y - pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot
    return mse, r2


def mc_dropout_predict(model: nn.Module, X: torch.Tensor, T: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo dropout BNN approximation: predictive mean and std."""
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(model(X))
    preds = torch.stack(preds, dim=0)
    return preds.mean(dim=0), preds.std(dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log(f"Device: {DEVICE}")
    p, d, d_clean = make_etch_dataset(n=320, noise=0.04)
    rgb = etch_to_rgb(d)

    # 80/20 train/test split
    perm = torch.randperm(p.shape[0], device=DEVICE)
    n_train = int(0.8 * p.shape[0])
    tr, te = perm[:n_train], perm[n_train:]
    X_tr_p, X_te_p = p[tr], p[te]
    X_tr_rgb, X_te_rgb = rgb[tr], rgb[te]
    y_tr, y_te = d[tr], d[te]
    y_te_clean = d_clean[te]

    # -------------------- (a) Linear baseline --------------------
    Xb_tr = torch.cat([X_tr_p, torch.ones(X_tr_p.shape[0], 1, device=DEVICE)], dim=1)
    Xb_te = torch.cat([X_te_p, torch.ones(X_te_p.shape[0], 1, device=DEVICE)], dim=1)
    w, *_ = torch.linalg.lstsq(Xb_tr, y_tr.unsqueeze(-1))
    pred_lin = (Xb_te @ w).squeeze(-1)
    mse_lin = F.mse_loss(pred_lin, y_te).item()
    log(f"\nLinear baseline (process params -> etch depth):  MSE={mse_lin:.3f} nm^2")

    # -------------------- (b) ANN on process params --------------------
    ann = ANN(in_dim=3, hidden=64, depth=3, dropout=0.0).to(DEVICE)
    train_regressor(ann, X_tr_p, y_tr)
    mse_a, r2_a = eval_metrics(ann, X_te_p, y_te)
    log(f"ANN (process params):                            MSE={mse_a:.3f}  R^2={r2_a:.3f}")

    # -------------------- (c) BNN (MC-dropout) on process params --------------------
    bnn = ANN(in_dim=3, hidden=64, depth=3, dropout=0.1).to(DEVICE)
    train_regressor(bnn, X_tr_p, y_tr, steps=2000)
    mu_te, sd_te = mc_dropout_predict(bnn, X_te_p, T=200)
    mse_b = F.mse_loss(mu_te, y_te).item()
    ss_res = ((y_te - mu_te) ** 2).sum().item()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
    r2_b = 1.0 - ss_res / ss_tot
    # 95 % credible interval coverage
    lo = mu_te - 1.96 * sd_te
    hi = mu_te + 1.96 * sd_te
    cov = ((y_te >= lo) & (y_te <= hi)).float().mean().item()
    avg_sd = sd_te.mean().item()
    log(
        f"BNN MC-dropout (process params):                 MSE={mse_b:.3f}  "
        f"R^2={r2_b:.3f}  avg sigma={avg_sd:.2f} nm  95% coverage={cov*100:.1f}%"
    )

    # -------------------- (d) DIC-only ANN --------------------
    dic_ann = ANN(in_dim=3, hidden=64, depth=3, dropout=0.0).to(DEVICE)
    train_regressor(dic_ann, X_tr_rgb, y_tr, steps=2000)
    mse_d, r2_d = eval_metrics(dic_ann, X_te_rgb, y_te)
    log(f"ANN (RGB / DIC only, no process parameters):     MSE={mse_d:.3f}  R^2={r2_d:.3f}")

    # ------------- noise floor for context ----------------
    noise_floor = F.mse_loss(y_te, y_te_clean).item()
    log(f"\n(Aleatoric noise floor on test set:              MSE={noise_floor:.3f} nm^2)")

    # ----- ranking summary -----
    log("\nSummary (lower MSE / higher R^2 is better):")
    log(f"  linear baseline             MSE={mse_lin:8.3f}")
    log(f"  ANN (process params)        MSE={mse_a:8.3f}  R2={r2_a:.3f}")
    log(f"  BNN MC-dropout              MSE={mse_b:8.3f}  R2={r2_b:.3f}  cov95={cov*100:.1f}%")
    log(f"  ANN (DIC RGB only)          MSE={mse_d:8.3f}  R2={r2_d:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
