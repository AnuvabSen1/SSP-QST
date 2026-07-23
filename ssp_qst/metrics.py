"""
metrics.py -- Evaluation metrics

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

- fidelity: Uhlmann fidelity F(rho, sigma).
- jz_operator / qfi: collective-spin generator and quantum Fisher
  information (Eq. 5), used for the QFI-agreement score of Fig. 5a.

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import numpy as np
from scipy.linalg import sqrtm
from .paulis import I2, Z, kron_all
from .states import project_density

def fidelity(rho, sigma):
    sr = sqrtm(rho)
    middle = sr @ sigma @ sr
    val = np.trace(sqrtm(middle))
    out = float(np.real(val * np.conj(val)))
    return min(max(out, 0.0), 1.0)

def jz_operator(n):
    ops = []
    for q in range(n):
        mats = [I2] * n
        mats[q] = Z
        ops.append(kron_all(mats))
    return 0.5 * sum(ops)

def qfi(rho, H, eps=1e-12):
    rho = project_density(rho)
    vals, vecs = np.linalg.eigh(rho)
    H_eig = vecs.conj().T @ H @ vecs
    F = 0.0
    for i in range(len(vals)):
        for j in range(len(vals)):
            denom = vals[i] + vals[j]
            if denom > eps:
                F += 2.0 * ((vals[i] - vals[j]) ** 2 / denom) * abs(H_eig[i, j]) ** 2
    return float(np.real(F))

# ------------------
# Style helpers
# ------------------
