"""
Reference implementation of a physics-aware DeepONet (PPAN) for plasma sheath
dynamics, including:
  * branch network on reactor controls (p, P_RF, f, gas-mix)
  * trunk network on (x [, t])
  * sheath/bulk gate that switches between two heads
  * physics-informed losses: Bohm criterion + ambipolar flux

This is a minimum-viable skeleton. Replace the synthetic generators below with
EEDF/fluid simulator outputs (e.g. HPEM, COMSOL Plasma Module).

Run: python ppan_deeponet_reference.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, out_dim, hidden=128, depth=4, act=nn.SiLU):
    layers = []
    last = in_dim
    for _ in range(depth):
        layers += [nn.Linear(last, hidden), act()]
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PPAN(nn.Module):
    """Physics-aware DeepONet with sheath/bulk routing."""

    def __init__(self, n_controls=4, n_basis=64, n_outputs=3):
        super().__init__()
        self.branch = mlp(n_controls, n_basis)
        self.trunk = mlp(1, n_basis)                  # use 2 if (x,t)
        self.gate = mlp(n_basis * 2, 1)               # logits: bulk vs sheath
        self.head_bulk = mlp(n_basis * 2, n_outputs)
        self.head_sheath = mlp(n_basis * 2, n_outputs)

    def forward(self, controls, x):
        b = self.branch(controls)                     # [B, n_basis]
        t = self.trunk(x)                             # [N, n_basis]
        # outer product features (DeepONet style)
        feat = b.unsqueeze(1) * t.unsqueeze(0)        # [B, N, n_basis]
        feat_cat = torch.cat([
            b.unsqueeze(1).expand(-1, t.shape[0], -1),
            t.unsqueeze(0).expand(b.shape[0], -1, -1),
        ], dim=-1)
        gate = torch.sigmoid(self.gate(feat_cat))
        y_bulk = self.head_bulk(feat_cat)
        y_sheath = self.head_sheath(feat_cat)
        return gate * y_sheath + (1 - gate) * y_bulk, gate.squeeze(-1)


# ------------------------ Physics-informed losses ------------------------
def bohm_penalty(u_i_at_edge, c_s):
    """Penalise u_i < c_s at sheath edge (Bohm criterion)."""
    return F.relu(c_s - u_i_at_edge).mean()


def ambipolar_penalty(gamma_e, gamma_i):
    return ((gamma_e - gamma_i) ** 2).mean()


# ------------------------ Synthetic data generator ------------------------
def synthetic_batch(B, N, device="cpu"):
    """Toy generator: pretend outputs are smoothed step + bulk linear profile."""
    controls = torch.rand(B, 4, device=device)            # p, P_RF, f, mix
    x = torch.linspace(0, 1, N, device=device).unsqueeze(-1)
    sheath_mask = (x.squeeze(-1) > 0.85).float()
    Vs = 30 + 50 * controls[:, 1:2]
    s = 0.05 + 0.05 * controls[:, 0:1]
    target_V = (1 - sheath_mask) * 0.0 + sheath_mask * Vs        # rough
    target_n = (1 - sheath_mask) * 1e16 + sheath_mask * 1e15
    target_T = 3.0 + 0.0 * x.T                                   # ~3 eV
    y = torch.stack([target_V.expand(B, N), target_n.expand(B, N), target_T.expand(B, N)], dim=-1)
    return controls, x, y


# ------------------------ Train / fine-tune loop ------------------------
def train(model, n_steps=200, lr=1e-3, src=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        controls, x, y = synthetic_batch(B=8 if src else 4, N=64)
        y_hat, gate = model(controls, x)
        data_loss = F.mse_loss(y_hat, y)
        # toy physics terms
        bohm = bohm_penalty(u_i_at_edge=torch.tensor(1.05), c_s=torch.tensor(1.0))
        amb = ambipolar_penalty(torch.tensor(1.0), torch.tensor(1.0))
        loss = data_loss + 0.1 * bohm + 0.1 * amb
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            mode = "PRETRAIN(CCP)" if src else "FINETUNE(ICP)"
            print(f"[{mode}] step {step:4d}  loss {loss.item():.4e}  gate-mean {gate.mean().item():.2f}")


if __name__ == "__main__":
    model = PPAN()
    print("Source pretraining on CCP-like data ...")
    train(model, n_steps=200, src=True)
    print("\nTransfer to ICP target with only a few samples ...")
    train(model, n_steps=50, lr=2e-4, src=False)
