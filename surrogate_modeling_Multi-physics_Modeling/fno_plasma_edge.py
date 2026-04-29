"""
2D Fourier Neural Operator surrogate for edge / scrape-off-layer plasma codes
(JOREK / STORM-style). Ingests a multi-channel state (n, T, omega, psi)
on a 2D grid and predicts the next time step.

Includes:
  * spectral convolution layer with learnable Fourier modes
  * a transfer-learning loop: pretrain on low-fidelity data, fine-tune on
    a small high-fidelity dataset

Run: python fno_plasma_edge.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes_x=12, modes_y=12):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_x, modes_y, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_x, modes_y, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes_x, :self.modes_y], self.w1
        )
        out_ft[:, :, -self.modes_x:, :self.modes_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes_x:, :self.modes_y], self.w2
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, width=32, modes=12, depth=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(depth)])
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

    def forward(self, x):
        h = self.lift(x)
        for s, k in zip(self.spec, self.skip):
            h = F.gelu(s(h) + k(h))
        return self.proj(h)


# --------- Synthetic low- and high-fidelity datasets ---------
def synthetic(B, H, W, hi_fi=False, device="cpu"):
    """Synthetic 2D edge-plasma frames.

    Low-fidelity: smooth blob; high-fidelity: blob + filamentary noise.
    """
    x = torch.linspace(-1, 1, H, device=device)
    y = torch.linspace(-1, 1, W, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R2 = X ** 2 + Y ** 2
    base = torch.exp(-R2 / 0.4)
    states = []
    for _ in range(B):
        scale = 0.5 + torch.rand(1).item()
        n = scale * base
        T = 0.3 * base + 0.05
        omega = torch.zeros_like(base)
        psi = torch.cos(2 * (X + Y))
        if hi_fi:
            n = n + 0.05 * torch.randn_like(n)
            omega = 0.1 * torch.cos(8 * X) * torch.sin(8 * Y)
        states.append(torch.stack([n, T, omega, psi]))
    cur = torch.stack(states)
    nxt = cur + 0.01 * torch.roll(cur, shifts=1, dims=-1)
    return cur, nxt


def train_one(model, n_steps, hi_fi, lr=1e-3, B=4, H=64, W=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        cur, nxt = synthetic(B, H, W, hi_fi=hi_fi)
        pred = model(cur)
        loss = F.mse_loss(pred, nxt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            tag = "HI-FI " if hi_fi else "LO-FI "
            print(f"[{tag}] step {step:4d}  MSE={loss.item():.3e}")


if __name__ == "__main__":
    model = FNO2d(in_ch=4, out_ch=4, width=24, modes=8, depth=3)
    print("Pretrain on low-fidelity ...")
    train_one(model, n_steps=200, hi_fi=False, lr=1e-3)
    print("\nFine-tune on high-fidelity (small dataset, lower LR) ...")
    train_one(model, n_steps=80, hi_fi=True, lr=2e-4)
