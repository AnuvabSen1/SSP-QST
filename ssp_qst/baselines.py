"""
baselines.py -- Non-iterative baselines

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

The two non-iterative baselines compared against SSP-QST:
- spectral_squaring: rho^2 / tr(rho^2), sharpens the spectrum but can
  distort relative signal weights.
- top_eigvec: rank-1 projection onto the dominant eigenvector, exact for
  pure probes and catastrophic for mixed ranks (Fig. 3).
Iterative maximum-likelihood baselines (RrhoR, diluted RrhoR) live in
experiments/run_ml_comparison.py.

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import numpy as np
from .states import density, project_density

def spectral_squaring(rho):
    r2 = rho @ rho
    return project_density(r2 / np.trace(r2))

def top_eigvec(rho):
    vals, vecs = np.linalg.eigh(project_density(rho))
    return density(vecs[:, -1])
