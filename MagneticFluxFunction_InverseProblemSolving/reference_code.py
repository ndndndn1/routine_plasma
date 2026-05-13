"""
Reference implementation of Zhou & Zhu (2025), arXiv:2507.16636 --
"Physics-Informed Neural Networks for High-Precision Grad-Shafranov Equilibrium
Reconstruction" -- focusing on their core contribution: a two-stage PINN that
drives the GS residual several orders of magnitude lower than a single-stage PINN.

We use a closed-form Solov'ev test case for verification:
    psi_an(R, Z) = (R^2 - 1)^2 / 8 + Z^2 / 2.
A direct calculation gives
    Delta* psi_an = R^2 + 1,
which fixes the source term used in the Grad-Shafranov residual. This pair is the
simplest Solov'ev equilibrium with a non-trivial spatial source, and serves the same
verification role as the Solov'ev profiles used in the paper.

Two-stage PINN protocol:
  Stage 1: train a network psi_1(R, Z) to minimize the GS residual + boundary loss.
  Stage 2: freeze psi_1, train a small correction psi_2(R, Z) that, when added to
           psi_1, drives the residual further down. The stage-2 loss is the SAME GS
           residual evaluated on (psi_1 + psi_2), but because the bulk of the
           solution is already in psi_1, stage 2 only has to fit the residual error.

Dependencies: numpy, torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Domain.
R0 = 1.0
R_LO, R_HI = 0.6, 1.4
Z_LO, Z_HI = -0.4, 0.4


def psi_analytic(R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Solov'ev test case: psi = (R^2 - 1)^2 / 8 + Z^2 / 2."""
    return (R**2 - R0**2) ** 2 / 8.0 + Z**2 / 2.0


def gs_source(R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """The right-hand side of Delta* psi_an = R^2 + 1 (with R0=1)."""
    return R**2 + 1.0


def grad_shafranov_residual(psi_func, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Residual r = Delta* psi - source, with Delta* the GS operator."""
    R = R.detach().requires_grad_(True)
    Z = Z.detach().requires_grad_(True)
    psi = psi_func(R, Z)
    psi_R = torch.autograd.grad(psi, R, torch.ones_like(psi), create_graph=True)[0]
    psi_RR = torch.autograd.grad(psi_R, R, torch.ones_like(psi_R), create_graph=True)[0]
    psi_Z = torch.autograd.grad(psi, Z, torch.ones_like(psi), create_graph=True)[0]
    psi_ZZ = torch.autograd.grad(psi_Z, Z, torch.ones_like(psi_Z), create_graph=True)[0]
    delta_star = psi_RR - psi_R / R + psi_ZZ
    return delta_star - gs_source(R, Z)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class PsiMLP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 5):
        super().__init__()
        seq = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*seq)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, R, Z):
        x = torch.stack([R.reshape(-1), Z.reshape(-1)], dim=-1)
        return self.net(x).squeeze(-1).reshape(R.shape)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_interior(n: int):
    R = torch.rand(n) * (R_HI - R_LO) + R_LO
    Z = torch.rand(n) * (Z_HI - Z_LO) + Z_LO
    return R, Z


def sample_boundary(n_per_side: int):
    s = torch.linspace(0, 1, n_per_side)
    R1 = torch.full_like(s, R_LO); Z1 = Z_LO + s * (Z_HI - Z_LO)
    R2 = torch.full_like(s, R_HI); Z2 = Z_LO + s * (Z_HI - Z_LO)
    R3 = R_LO + s * (R_HI - R_LO); Z3 = torch.full_like(s, Z_LO)
    R4 = R_LO + s * (R_HI - R_LO); Z4 = torch.full_like(s, Z_HI)
    R = torch.cat([R1, R2, R3, R4]); Z = torch.cat([Z1, Z2, Z3, Z4])
    return R, Z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_stage(stage_net: nn.Module, steps: int, lr: float, tag: str,
                base_net: nn.Module | None = None,
                bc_weight: float = 50.0,
                lbfgs_finetune: int = 0) -> None:
    """If base_net is given, train stage_net as a correction added to base_net.
    If lbfgs_finetune > 0, run an L-BFGS polish at the end (paper-style refinement)."""
    opt = torch.optim.Adam(stage_net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    if base_net is None:
        def total(R, Z): return stage_net(R, Z)
    else:
        def total(R, Z): return base_net(R, Z) + stage_net(R, Z)

    for step in range(steps):
        opt.zero_grad()
        R, Z = sample_interior(2048)
        Rb, Zb = sample_boundary(64)

        res = grad_shafranov_residual(total, R, Z)
        loss_pde = (res**2).mean()

        psi_b_pred = total(Rb, Zb)
        psi_b_true = psi_analytic(Rb, Zb)
        loss_bc = ((psi_b_pred - psi_b_true) ** 2).mean()

        loss = loss_pde + bc_weight * loss_bc
        loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % max(1, steps // 5) == 0:
            with torch.no_grad():
                R_e, Z_e = sample_interior(4096)
                err = (total(R_e, Z_e) - psi_analytic(R_e, Z_e)).abs().mean().item()
            print(f"  [{tag}] step {step + 1:05d}  "
                  f"loss_pde={loss_pde.item():.3e}  "
                  f"loss_bc={loss_bc.item():.3e}  "
                  f"|psi - psi_true|_mean={err:.3e}")

    if lbfgs_finetune > 0:
        # Fixed (non-stochastic) batch for L-BFGS.
        R_lbfgs, Z_lbfgs = sample_interior(8192)
        Rb_lbfgs, Zb_lbfgs = sample_boundary(256)
        lbfgs = torch.optim.LBFGS(stage_net.parameters(), lr=1.0,
                                  max_iter=lbfgs_finetune, history_size=50,
                                  tolerance_grad=1e-12, tolerance_change=1e-15,
                                  line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            res = grad_shafranov_residual(total, R_lbfgs, Z_lbfgs)
            loss_pde = (res**2).mean()
            loss_bc = ((total(Rb_lbfgs, Zb_lbfgs)
                        - psi_analytic(Rb_lbfgs, Zb_lbfgs)) ** 2).mean()
            l = loss_pde + bc_weight * loss_bc
            l.backward()
            return l

        lbfgs.step(closure)
        with torch.no_grad():
            R_e, Z_e = sample_interior(4096)
            err = (total(R_e, Z_e) - psi_analytic(R_e, Z_e)).abs().mean().item()
        print(f"  [{tag}] after L-BFGS polish  |psi - psi_true|_mean={err:.3e}")


def evaluate(psi_func) -> float:
    with torch.no_grad():
        R_e, Z_e = sample_interior(40000)
        err = (psi_func(R_e, Z_e) - psi_analytic(R_e, Z_e)).abs().mean()
    return float(err)


def main():
    torch.manual_seed(0)

    print("Stage 1: vanilla PINN (Adam)")
    stage1 = PsiMLP(hidden=64, layers=5)
    train_stage(stage1, steps=3000, lr=3e-3, tag="stage1")
    err1 = evaluate(lambda R, Z: stage1(R, Z))
    print(f"  >>> Stage 1 mean |psi error| = {err1:.3e}\n")

    print("Stage 2: correction network on top of stage 1 (frozen), Adam + L-BFGS polish")
    for p in stage1.parameters():
        p.requires_grad_(False)
    stage2 = PsiMLP(hidden=64, layers=5)
    train_stage(stage2, steps=2000, lr=1e-3, tag="stage2", base_net=stage1,
                lbfgs_finetune=300)
    err2 = evaluate(lambda R, Z: stage1(R, Z) + stage2(R, Z))
    print(f"  >>> Stage 2 mean |psi error| = {err2:.3e}")
    print(f"  >>> Error reduction factor   = {err1 / max(err2, 1e-30):.2f}x")


if __name__ == "__main__":
    main()
