"""
purification.py -- SSP-QST: Spectral Subspace Purification (Algorithm 1)

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

The method of the paper. Given a least-squares estimate rho_LS and
the shot count N_s:
  1. Eigendecompose the Hermitised estimate.
  2. Estimate the depolarising-equivalent noise level p_hat = 1 - lambda_max.
  3. Weyl-motivated noise floor: eps_th = p_hat/(d-1) + 0.5/sqrt(N_s) (Eq. 6).
  4. Rank-1 override: if lambda_{d-1} < 2 eps_th, return the top eigenvector.
  5. Zero eigenvalues below eps_th, renormalise, reconstruct.
Closed form, one eigendecomposition, O(d^3), no rank prior.

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import numpy as np
from .states import project_density

def ssp_qst(rho_ls, Ns=4096, return_rank=False):
    rho = project_density(rho_ls)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals, 0)
    vals /= max(vals.sum(), 1e-15)
    d = rho.shape[0]
    p_hat = max(0.0, 1.0 - vals[-1])
    eps_th = p_hat / (d - 1) + 0.5 / np.sqrt(Ns)
    if d > 1 and vals[-2] < 2 * eps_th:
        out = np.zeros_like(vals); out[-1] = 1.0
        rank = 1
    else:
        mask = vals > eps_th
        rank = int(np.sum(mask))
        out = vals * mask
        if out.sum() <= 1e-15:
            out[-1] = 1.0
            rank = 1
        out /= out.sum()
    rho_out = project_density(vecs @ np.diag(out) @ vecs.conj().T)
    return (rho_out, rank) if return_rank else rho_out
