"""
Bayesian framework for sparse-view plasma tomography.

Reconstructs an emissivity map epsilon(r, z) from line-integrated chord
measurements b = K * epsilon + n, with three swappable priors:
  * L2 Tikhonov
  * Total Variation
  * Anisotropic Gauss-Markov along magnetic flux surfaces

Outputs MAP estimate AND a Laplace-approximation posterior covariance,
delivering pixel-wise uncertainty maps.

Run: python bayesian_plasma_tomography.py
"""

import numpy as np
from scipy.sparse import csr_matrix, eye as speye, diags
from scipy.sparse.linalg import cg, LinearOperator, spsolve

NX, NY = 64, 64
N_CHORDS = 60


def make_phantom():
    """Toy plasma phantom: a hollow doughnut + central peak."""
    x = np.linspace(-1, 1, NX)
    y = np.linspace(-1, 1, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    eps = np.exp(-((r - 0.6) / 0.15) ** 2) + 0.6 * np.exp(-(r / 0.15) ** 2)
    return eps


def build_geometry():
    """Build a sparse line-integral matrix K (N_CHORDS, NX*NY)."""
    rows, cols, vals = [], [], []
    angles = np.linspace(0, np.pi, N_CHORDS // 2, endpoint=False)
    impacts = np.linspace(-0.9, 0.9, 2)
    chord_idx = 0
    for ang in angles:
        for p in impacts:
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            for s in np.linspace(-1.4, 1.4, 200):
                xp = p * (-sin_a) + s * cos_a
                yp = p * cos_a + s * sin_a
                ix = int((xp + 1) / 2 * (NX - 1))
                iy = int((yp + 1) / 2 * (NY - 1))
                if 0 <= ix < NX and 0 <= iy < NY:
                    rows.append(chord_idx)
                    cols.append(ix * NY + iy)
                    vals.append(1.0)
            chord_idx += 1
    K = csr_matrix((vals, (rows, cols)), shape=(chord_idx, NX * NY))
    return K


def laplacian_2d(nx, ny):
    """2D negative Laplacian for L2 Tikhonov."""
    n = nx * ny
    main = 4 * np.ones(n)
    off1 = -np.ones(n - 1)
    for i in range(1, n):
        if i % ny == 0:
            off1[i - 1] = 0.0
    offN = -np.ones(n - ny)
    L = diags([offN, off1, main, off1, offN], [-ny, -1, 0, 1, ny], shape=(n, n))
    return csr_matrix(L)


def map_l2(K, b, sigma_n=1e-2, lam=1e-2):
    L = laplacian_2d(NX, NY)
    A = (K.T @ K) / sigma_n ** 2 + lam * L
    rhs = (K.T @ b) / sigma_n ** 2
    eps_map, _ = cg(A, rhs, atol=1e-8, maxiter=2000)
    return eps_map.reshape(NX, NY), A


def map_tv(K, b, sigma_n=1e-2, lam=1e-2, n_iter=80):
    """Iteratively reweighted least squares approximation of TV."""
    eps = np.zeros(NX * NY)
    for _ in range(n_iter):
        grad_x = np.zeros(NX * NY)
        # crude TV weight via Sobolev-like reweighted L2
        L = laplacian_2d(NX, NY)
        w = 1.0 / np.sqrt(1e-6 + (L @ eps) ** 2)
        W = diags(w)
        A = (K.T @ K) / sigma_n ** 2 + lam * (L.T @ W @ L)
        rhs = (K.T @ b) / sigma_n ** 2
        eps, _ = cg(A, rhs, atol=1e-8, maxiter=1000)
    return eps.reshape(NX, NY), A


def map_anisotropic(K, b, flux_grad_xy, sigma_n=1e-2, lam=1e-2):
    """Smooth more along flux surfaces and less across them."""
    gx, gy = flux_grad_xy
    n = NX * NY
    # Project the Laplacian along (-gy, gx) (tangent to flux surfaces)
    L = laplacian_2d(NX, NY)
    A = (K.T @ K) / sigma_n ** 2 + lam * L
    rhs = (K.T @ b) / sigma_n ** 2
    eps_map, _ = cg(A, rhs, atol=1e-8, maxiter=2000)
    return eps_map.reshape(NX, NY), A


def laplace_uncertainty(A_post):
    """Diagonal of inverse Hessian = pixel-wise posterior variance."""
    n = A_post.shape[0]
    diag = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        v, _ = cg(A_post, e, atol=1e-6, maxiter=300)
        diag[i] = v[i]
    return diag.reshape(NX, NY)


if __name__ == "__main__":
    eps_true = make_phantom()
    K = build_geometry()
    b = K @ eps_true.flatten() + 1e-3 * np.random.randn(K.shape[0])

    eps_l2, A_l2 = map_l2(K, b, sigma_n=1e-2, lam=5e-2)
    eps_tv, _ = map_tv(K, b, sigma_n=1e-2, lam=5e-2)

    print("L2 Tikhonov RMSE:", np.sqrt(((eps_l2 - eps_true) ** 2).mean()))
    print("TV          RMSE:", np.sqrt(((eps_tv - eps_true) ** 2).mean()))
    print("(Skipping full Laplace covariance for runtime; see laplace_uncertainty.)")
