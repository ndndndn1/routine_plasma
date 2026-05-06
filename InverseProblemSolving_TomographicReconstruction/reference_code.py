"""Reference implementation of the *inverse rendering* idea from
Oezturk et al., "Inverse Rendering of Fusion Plasmas: Inferring Plasma
Composition from Imaging Systems" (arXiv:2408.07555 / Nuclear Fusion 2025).

The full paper builds a differentiable raytracer on top of Mitsuba 3 with
Null-Scattering and Path-Replay Backpropagation. Here we strip the idea down
to its essentials and implement it in pure PyTorch on a 2D poloidal slice:

    * a ground-truth poloidal map (n_n, n_e, T_e) is constructed on a grid;
    * a *forward* model line-integrates an atomic emissivity epsilon(R,Z)
      along a fan of camera rays to produce a 1-D synthetic image;
    * the inverse problem is solved by gradient descent through this
      differentiable line integral, recovering n_n(R,Z) from the synthetic
      image alone (n_e, T_e held fixed).

This is the simplest non-trivial demonstration that "image -> poloidal field"
inverse rendering of fusion plasmas works as a differentiable optimisation.

Run:
    pip install numpy torch
    python reference_code.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Geometry of the poloidal slice
# ---------------------------------------------------------------------------
NR, NZ = 64, 64                   # grid resolution
R_MIN, R_MAX = 0.4, 1.6           # major radius range [m]
Z_MIN, Z_MAX = -0.6, 0.6          # vertical range [m]
R_AXIS, Z_AXIS = 1.0, 0.0
A_MIN = 0.45                      # plasma minor radius [m]


def grid_RZ() -> tuple[torch.Tensor, torch.Tensor]:
    R = torch.linspace(R_MIN, R_MAX, NR, device=DEVICE)
    Z = torch.linspace(Z_MIN, Z_MAX, NZ, device=DEVICE)
    Rg, Zg = torch.meshgrid(R, Z, indexing="ij")
    return Rg, Zg


def rho(Rg: torch.Tensor, Zg: torch.Tensor) -> torch.Tensor:
    """Normalised flux-like coordinate (circular flux surfaces)."""
    return torch.sqrt(((Rg - R_AXIS) ** 2 + Zg**2)) / A_MIN


def make_truth() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic ground-truth poloidal maps for n_e, T_e, n_n in *normalised*
    units so the emissivity stays in machine range. Magnitudes are not the
    point — the inverse-rendering principle is.

    n_e: peaked profile ~ (1 - rho^2)^a inside the plasma. Units: arb.
    T_e: similar peaked profile (1 unit ~ 1 keV).
    n_n: edge-localised neutrals shell with a poloidal asymmetry (HFS pump-out
         + a small lower X-point bright spot) — this is the unknown to recover.
    """
    Rg, Zg = grid_RZ()
    r = rho(Rg, Zg)
    inside = (r <= 1.0).float()
    n_e = inside * (1.0 - r.clamp_max(1.0) ** 2) ** 1.2 + 0.01
    T_e = inside * (1.0 - r.clamp_max(1.0) ** 2) ** 0.8 * 1.0 + 0.01
    # neutrals shell at the edge plus an X-point spot near (R0, -0.4)
    edge = torch.exp(-((r - 1.0) ** 2) / (2 * 0.06**2))
    asym = 0.6 + 0.4 * torch.tanh(2.0 * (R_AXIS - Rg))
    xpt = torch.exp(-(((Rg - R_AXIS) ** 2 + (Zg + 0.4) ** 2)) / (2 * 0.06**2))
    n_n = (1.0 * edge * asym + 4.0 * xpt).clamp_min(1e-3)
    return n_e, T_e, n_n


# ---------------------------------------------------------------------------
# Atomic emissivity model (D-alpha-like Boltzmann factor)
#   eps(R,Z)  ~  n_n * n_e * sqrt(T_e) * exp(-E0 / T_e)
# Constants are arbitrary scaling.
# ---------------------------------------------------------------------------
E0 = 0.5    # excitation energy in same units as T_e (~ keV)


def emissivity(n_n: torch.Tensor, n_e: torch.Tensor, T_e: torch.Tensor) -> torch.Tensor:
    Te = torch.clamp(T_e, min=1e-2)
    return n_n * n_e * torch.sqrt(Te) * torch.exp(-E0 / Te)


# ---------------------------------------------------------------------------
# Differentiable line-of-sight renderer (2D)
# A fan of N_PIX rays from a virtual camera pinhole sweeps the slice. Each ray
# is sampled at N_STEPS points; emissivity along the ray is linearly
# interpolated (bilinear) from the 2D grid via grid_sample (autograd-able).
# ---------------------------------------------------------------------------
N_PIX = 96
N_STEPS = 200
FOV = math.radians(60.0)
# Four virtual camera views around the plasma (LFS, top, HFS-like, bottom)
CAMERAS: list[tuple[float, float]] = [
    (R_MAX + 0.6, 0.0),        # outboard midplane
    (R_AXIS, Z_MAX + 0.6),     # top
    (R_MIN - 0.4, 0.0),        # inboard midplane (looking outwards)
    (R_AXIS, Z_MIN - 0.6),     # bottom
]


def make_rays_for(cam_R: float, cam_Z: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (n_pix, n_steps) sample points (R,Z) for one camera view."""
    angles = torch.linspace(-FOV / 2, FOV / 2, N_PIX, device=DEVICE)
    # central look direction = from camera toward plasma centre (R_AXIS, Z_AXIS)
    dx_c = R_AXIS - cam_R
    dz_c = Z_AXIS - cam_Z
    norm = math.hypot(dx_c, dz_c)
    dx_c, dz_c = dx_c / norm, dz_c / norm
    # rotate central direction by each angle
    cosA = torch.cos(angles)
    sinA = torch.sin(angles)
    dx = dx_c * cosA - dz_c * sinA
    dz = dx_c * sinA + dz_c * cosA
    # path length covers ~2x distance to centre to fully traverse plasma
    Lmax = 2.0 * norm + 0.4
    t = torch.linspace(0.0, Lmax, N_STEPS, device=DEVICE)
    Rs = cam_R + dx[:, None] * t[None, :]
    Zs = cam_Z + dz[:, None] * t[None, :]
    return Rs, Zs


def make_rays() -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ray samples across all camera views."""
    Rs_all, Zs_all = [], []
    for cR, cZ in CAMERAS:
        Rs, Zs = make_rays_for(cR, cZ)
        Rs_all.append(Rs)
        Zs_all.append(Zs)
    return torch.cat(Rs_all, dim=0), torch.cat(Zs_all, dim=0)


def sample_grid(field: torch.Tensor, Rs: torch.Tensor, Zs: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a (NR, NZ) field at points (Rs, Zs) in physical units.

    Uses torch.nn.functional.grid_sample which is fully differentiable.
    """
    # Normalise to grid_sample convention: x in [-1, 1] for last dim, y for second-to-last.
    # field has shape (NR, NZ); grid_sample expects (N, C, H, W) with H=NZ, W=NR for our (R,Z) layout.
    f = field.t().contiguous().unsqueeze(0).unsqueeze(0)   # (1,1,NZ,NR)
    R_norm = 2 * (Rs - R_MIN) / (R_MAX - R_MIN) - 1
    Z_norm = 2 * (Zs - Z_MIN) / (Z_MAX - Z_MIN) - 1
    grid = torch.stack([R_norm, Z_norm], dim=-1).unsqueeze(0)  # (1, NPIX, NSTEPS, 2)
    sampled = F.grid_sample(
        f, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled[0, 0]                                        # (NPIX, NSTEPS)


def render(n_n: torch.Tensor, n_e: torch.Tensor, T_e: torch.Tensor) -> torch.Tensor:
    """Forward render: stacked line-of-sight integrals over all camera views."""
    eps = emissivity(n_n, n_e, T_e)
    Rs, Zs = make_rays()
    eps_along_ray = sample_grid(eps, Rs, Zs)                    # (N_PIX*N_CAMS, NSTEPS)
    # path step length per ray (use a representative length)
    ds = (2.0 * math.hypot(R_MAX - R_MIN, Z_MAX - Z_MIN)) / (N_STEPS - 1)
    image = eps_along_ray.sum(dim=1) * ds                       # (N_PIX*N_CAMS,)
    return image


# ---------------------------------------------------------------------------
# Total-Variation regulariser (to suppress spurious high-frequency artefacts)
# ---------------------------------------------------------------------------
def tv(field: torch.Tensor) -> torch.Tensor:
    dx = field[1:, :] - field[:-1, :]
    dz = field[:, 1:] - field[:, :-1]
    return dx.abs().mean() + dz.abs().mean()


# ---------------------------------------------------------------------------
# Inverse problem: recover n_n(R,Z) from the noisy synthetic image
# ---------------------------------------------------------------------------
def main() -> None:
    log(f"Device: {DEVICE}")
    n_e, T_e, n_n_true = make_truth()
    img_clean = render(n_n_true, n_e, T_e).detach()
    # Poisson-like shot noise (simulates a real photon-counting camera)
    img_obs = img_clean + 0.03 * img_clean.abs().mean() * torch.randn_like(img_clean)
    log(f"image stats: min={img_obs.min().item():.3e} max={img_obs.max().item():.3e}")

    # Plasma support mask: known plasma boundary (rho <= 1.05). Outside this
    # region the neutrals are fixed to a small floor — this mirrors how real
    # diagnostics combine known geometry with imaging measurements.
    Rg, Zg = grid_RZ()
    support = (rho(Rg, Zg) <= 1.05).float()

    # The plasma neutrals problem is severely underdetermined if we leave one
    # unknown per pixel (4096 unknowns, only ~400 image pixels). We therefore
    # parameterise n_n on a coarse 12x12 control grid and bilinearly upsample
    # to 64x64 before rendering — a standard low-rank reduction that mirrors
    # the way the paper restricts the inverse problem to a smooth basis. The
    # optimisation now has 144 free parameters.
    NC = 12
    theta_coarse = torch.full((NC, NC), -2.0, device=DEVICE, requires_grad=True)
    n_n_max = 5.0

    def expand(theta_c: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(
            theta_c.unsqueeze(0).unsqueeze(0),
            size=(NR, NZ), mode="bilinear", align_corners=True,
        )[0, 0]
        return n_n_max * torch.sigmoid(up) * support

    n_steps = 1500
    opt = torch.optim.Adam([theta_coarse], lr=0.3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    img_norm = img_obs.norm() + 1e-12
    log("Starting gradient descent on coarse 12x12 control grid for n_n...")
    for step in range(n_steps):
        n_n_pred = expand(theta_coarse)
        img_pred = render(n_n_pred, n_e, T_e)
        loss_data = ((img_pred - img_obs) ** 2).sum() / img_norm**2
        loss_tv = tv(n_n_pred)
        loss = loss_data + 5e-3 * loss_tv
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 150 == 0:
            with torch.no_grad():
                rel = (n_n_pred - n_n_true).norm() / n_n_true.norm()
            log(
                f"  step {step+1:4d}  data_loss={loss_data.item():.3e}  "
                f"tv={loss_tv.item():.3e}  rel(n_n)={rel.item():.3e}"
            )

    with torch.no_grad():
        n_n_pred = expand(theta_coarse)
        n_n_init = expand(torch.full((NC, NC), -2.0, device=DEVICE))
        rel0 = (n_n_init - n_n_true).norm() / n_n_true.norm()
        rel1 = (n_n_pred - n_n_true).norm() / n_n_true.norm()
        img_final = render(n_n_pred, n_e, T_e)
        img_rel = (img_final - img_obs).norm() / img_obs.norm()
        log("\nFinal reconstruction quality:")
        log(f"  initial rel L2 error  ||n_n0 - n_n_true|| / ||n_n_true|| = {rel0.item():.3e}")
        log(f"  final   rel L2 error  ||n_n* - n_n_true|| / ||n_n_true|| = {rel1.item():.3e}")
        log(f"  re-rendered image vs observed:                               {img_rel.item():.3e}")
        log(
            f"  peak ground truth n_n: {n_n_true.max().item():.3e}, "
            f"peak recovered: {n_n_pred.max().item():.3e}"
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
