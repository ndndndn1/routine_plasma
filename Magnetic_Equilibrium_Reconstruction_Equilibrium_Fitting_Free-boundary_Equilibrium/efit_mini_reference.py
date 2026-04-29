"""
Simplified reference implementation of the EFIT-mini idea:
Multi-task NN -> profile/boundary parameters -> single Picard step
of the free-boundary Grad-Shafranov equation -> flux map psi(R,Z).

This is a teaching skeleton, not a production solver. It demonstrates:
  * encoding 68 magnetic measurements into a low-dim parameter vector
  * basis decomposition of p'(psi) and FF'(psi)
  * one Picard iteration with a precomputed Green's function inverse
    of the elliptic operator Delta* on a (R,Z) grid

Run: python efit_mini_reference.py
"""

import numpy as np
import torch
import torch.nn as nn

# ----------------------------- Grid -----------------------------
NR, NZ = 65, 65            # use 129x129 to match the paper
R_MIN, R_MAX = 0.2, 1.4
Z_MIN, Z_MAX = -0.8, 0.8
MU0 = 4 * np.pi * 1e-7

R = np.linspace(R_MIN, R_MAX, NR)
Z = np.linspace(Z_MIN, Z_MAX, NZ)
RR, ZZ = np.meshgrid(R, Z, indexing="ij")
DR = R[1] - R[0]
DZ = Z[1] - Z[0]


def build_delta_star_inverse():
    """Assemble Delta* on a 5-point stencil with Dirichlet BCs and invert.

    Delta* psi = R d/dR(1/R dpsi/dR) + d^2 psi/dZ^2 = -mu0 R J_phi
    """
    n = NR * NZ
    A = np.zeros((n, n))
    for i in range(NR):
        for j in range(NZ):
            k = i * NZ + j
            if i in (0, NR - 1) or j in (0, NZ - 1):
                A[k, k] = 1.0          # Dirichlet psi = 0 at vacuum vessel
                continue
            r = R[i]
            A[k, k] = -2 / DR ** 2 - 2 / DZ ** 2
            A[k, (i + 1) * NZ + j] = 1 / DR ** 2 - 1 / (2 * r * DR)
            A[k, (i - 1) * NZ + j] = 1 / DR ** 2 + 1 / (2 * r * DR)
            A[k, i * NZ + (j + 1)] = 1 / DZ ** 2
            A[k, i * NZ + (j - 1)] = 1 / DZ ** 2
    return np.linalg.inv(A)


# ----------------------------- Multi-task NN -----------------------------
class EFITMiniNet(nn.Module):
    """Maps 68 magnetic measurements to GS source parameters."""

    def __init__(self, n_basis=4):
        super().__init__()
        self.n_basis = n_basis
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(68, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # heads: p' coeffs, FF' coeffs, boundary descriptors (Raxis, Zaxis, kappa, delta)
        self.head_pprime = nn.Linear(hidden, n_basis)
        self.head_ffprime = nn.Linear(hidden, n_basis)
        self.head_boundary = nn.Linear(hidden, 4)

    def forward(self, x):
        h = self.trunk(x)
        return {
            "pprime": self.head_pprime(h),
            "ffprime": self.head_ffprime(h),
            "boundary": self.head_boundary(h),
        }


# ----------------------------- Source J_phi -----------------------------
def jphi_from_params(psi_n, pprime_coef, ffprime_coef, R_grid):
    """Toroidal current density from polynomial bases in normalised flux psi_n."""
    basis = np.stack([psi_n ** k for k in range(len(pprime_coef))], axis=-1)
    pprime = basis @ pprime_coef
    ffprime = basis @ ffprime_coef
    return R_grid * pprime + ffprime / (MU0 * R_grid)


def picard_step(psi, params, A_inv):
    psi_axis, psi_bnd = psi.min(), psi.max()
    psi_n = np.clip((psi - psi_axis) / (psi_bnd - psi_axis + 1e-9), 0.0, 1.0)
    j_phi = jphi_from_params(psi_n, params["pprime"], params["ffprime"], RR)
    rhs = (-MU0 * RR * j_phi).flatten()
    psi_new = (A_inv @ rhs).reshape(NR, NZ)
    return psi_new, j_phi


# ----------------------------- End-to-end pipeline -----------------------------
def reconstruct(measurements, model, A_inv, psi_init=None):
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(measurements, dtype=torch.float32).unsqueeze(0))
    params = {k: v.squeeze(0).numpy() for k, v in out.items()}
    psi = np.zeros((NR, NZ)) if psi_init is None else psi_init.copy()
    psi, j_phi = picard_step(psi, params, A_inv)
    return psi, j_phi, params


if __name__ == "__main__":
    A_inv = build_delta_star_inverse()
    model = EFITMiniNet()
    fake_measurements = np.random.randn(68).astype(np.float32) * 0.1
    psi, j_phi, params = reconstruct(fake_measurements, model, A_inv)
    print(f"psi map: shape={psi.shape}, min={psi.min():.3e}, max={psi.max():.3e}")
    print(f"J_phi:   shape={j_phi.shape}, integrated Ip~{(j_phi * DR * DZ).sum():.3e} A")
    print(f"boundary descriptors (Raxis,Zaxis,kappa,delta) ~ {params['boundary']}")
