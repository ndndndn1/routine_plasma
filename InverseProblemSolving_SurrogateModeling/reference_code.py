"""
Reference implementation inspired by:
  P. Curvo, D. R. Ferreira, R. Jorge, "Using Deep Learning to Design High
  Aspect Ratio Fusion Devices", J. Plasma Phys. 91 E38 (2025),
  arXiv:2409.00564.

Demonstrates the two-stage pattern used in the paper for inverse design
and, more broadly, for plasma inverse problems where several distinct
states can explain the same observations:

  1. Forward surrogate f_theta(x) ~ f(x), trained to replace the expensive
     (near-axis MHD) solver.
  2. Inverse Mixture Density Network p_phi(x | y), trained with Bishop's
     NLL so that mixture components cover all pre-images of a target y.

We replace the stellarator near-axis solver with a toy plasma-like forward
model whose map is deliberately many-to-one, so you can see the MDN resolve
ambiguity. No external stellarator libraries are required.

Run: python reference_code.py     (needs numpy, torch)
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RNG = np.random.default_rng(42)
DESIGN_DIM = 4      # pretend-Fourier coefficients of axis shape
TARGET_DIM = 3      # pretend figures-of-merit (iota, magnetic well, elongation)
K_MIX = 4           # mixture components in the MDN


def forward_physics(x: np.ndarray) -> np.ndarray:
    """Toy many-to-one forward model standing in for the near-axis solver."""
    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    # Squared terms -> sign ambiguity; trig terms -> periodic ambiguity.
    y1 = x1**2 + 0.5 * x2**2           # ambiguous in sign of x1, x2
    y2 = np.sin(math.pi * x3) + 0.3 * x4
    y3 = np.tanh(x1 * x2) + 0.2 * x4**2
    return np.stack([y1, y2, y3], axis=-1).astype(np.float32)


def make_dataset(n: int = 20000):
    x = RNG.uniform(-1.0, 1.0, size=(n, DESIGN_DIM)).astype(np.float32)
    y = forward_physics(x)
    y = y + 0.02 * RNG.standard_normal(y.shape).astype(np.float32)
    return x, y


class ForwardSurrogate(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DESIGN_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, TARGET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MDN(nn.Module):
    """Bishop Mixture Density Network: p(x | y) = sum_k pi_k N(x; mu_k, sigma_k)."""

    def __init__(self, hidden: int = 128, k: int = K_MIX):
        super().__init__()
        self.k = k
        self.trunk = nn.Sequential(
            nn.Linear(TARGET_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_pi = nn.Linear(hidden, k)
        self.head_mu = nn.Linear(hidden, k * DESIGN_DIM)
        self.head_log_sigma = nn.Linear(hidden, k * DESIGN_DIM)

    def forward(self, y: torch.Tensor):
        h = self.trunk(y)
        log_pi = torch.log_softmax(self.head_pi(h), dim=-1)          # (B, K)
        mu = self.head_mu(h).view(-1, self.k, DESIGN_DIM)            # (B, K, D)
        log_sigma = self.head_log_sigma(h).view(-1, self.k, DESIGN_DIM)
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=2.0)
        return log_pi, mu, log_sigma


def mdn_nll(x: torch.Tensor, log_pi: torch.Tensor, mu: torch.Tensor,
            log_sigma: torch.Tensor) -> torch.Tensor:
    # Diagonal Gaussian log-density per component, then log-sum-exp over K.
    x_ = x.unsqueeze(1)                                    # (B, 1, D)
    inv_var = torch.exp(-2.0 * log_sigma)
    log_det = log_sigma.sum(dim=-1)                        # (B, K)
    log_norm = -0.5 * DESIGN_DIM * math.log(2.0 * math.pi) - log_det
    quad = -0.5 * ((x_ - mu) ** 2 * inv_var).sum(dim=-1)   # (B, K)
    log_comp = log_norm + quad                             # (B, K)
    log_prob = torch.logsumexp(log_pi + log_comp, dim=-1)  # (B,)
    return -log_prob.mean()


def sample_from_mdn(mdn: MDN, y: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
    """Sample n_samples design points per target y."""
    with torch.no_grad():
        log_pi, mu, log_sigma = mdn(y)                     # (B, K), (B, K, D), (B, K, D)
        pi = log_pi.exp()
        b = y.shape[0]
        comp = torch.multinomial(pi, num_samples=n_samples, replacement=True)  # (B, S)
        mu_s = torch.gather(mu, 1, comp.unsqueeze(-1).expand(-1, -1, DESIGN_DIM))
        sig_s = torch.gather(torch.exp(log_sigma), 1,
                             comp.unsqueeze(-1).expand(-1, -1, DESIGN_DIM))
        eps = torch.randn_like(mu_s)
        return mu_s + sig_s * eps                          # (B, S, D)


def train_forward(x: np.ndarray, y: np.ndarray) -> ForwardSurrogate:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    model = ForwardSurrogate()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    for epoch in range(15):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(xb)
        if epoch % 3 == 0:
            print(f"[fwd] epoch {epoch:2d}  mse={total/len(ds):.4f}")
    return model


def train_mdn(x: np.ndarray, y: np.ndarray) -> MDN:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    model = MDN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(40):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            log_pi, mu, log_sigma = model(yb)
            loss = mdn_nll(xb, log_pi, mu, log_sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.detach().item() * len(xb)
        if epoch % 5 == 0:
            print(f"[mdn] epoch {epoch:2d}  nll={total/len(ds):.4f}")
    return model


def main() -> None:
    x, y = make_dataset(n=20000)
    fwd = train_forward(x, y)
    mdn = train_mdn(x, y)

    # Pick a random target y_target and ask: which x values realise it?
    y_target_np = y[:5]
    y_target = torch.from_numpy(y_target_np)

    samples = sample_from_mdn(mdn, y_target, n_samples=64)    # (5, 64, D)
    samples_np = samples.numpy()

    # Push candidates back through the true physics model and the surrogate.
    flat = samples_np.reshape(-1, DESIGN_DIM)
    y_true = forward_physics(flat).reshape(samples_np.shape[0], samples_np.shape[1], -1)
    y_surr = fwd(torch.from_numpy(flat)).detach().numpy().reshape(y_true.shape)

    # Report achieved-vs-target error and candidate diversity per target.
    for i in range(y_target_np.shape[0]):
        err_true = np.linalg.norm(y_true[i] - y_target_np[i], axis=-1).mean()
        err_surr = np.linalg.norm(y_surr[i] - y_target_np[i], axis=-1).mean()
        diversity = samples_np[i].std(axis=0).mean()
        print(
            f"target {i}: mean |y_true - y*|={err_true:.3f}  "
            f"mean |y_surr - y*|={err_surr:.3f}  "
            f"design-space std across 64 samples={diversity:.3f}"
        )


if __name__ == "__main__":
    main()
