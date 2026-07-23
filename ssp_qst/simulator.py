"""
simulator.py -- Noise channels and tomographic measurement simulator

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

The measurement simulator behind every experiment:
- depolarize: depolarising channel rho -> (1-p) rho + p I/d.
- ls_qst_from_state: linear-inversion (least-squares) QST from Pauli
  expectations with analytic Bernoulli shot noise at N_s shots per
  setting (Eqs. 3-4 of the paper). Deterministic under a fixed seed.

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import numpy as np
from .purification import ssp_qst

from .paulis import pauli_basis
from .states import project_density, parameterized_ghz_density

def depolarize(rho, p):
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d

def ls_qst_from_state(rho_true, n, Ns=4096, seed=0):
    rng = np.random.default_rng(seed)
    d = 2 ** n
    rho_raw = np.zeros((d, d), dtype=complex)
    for P in pauli_basis(n):
        mu = float(np.real(np.trace(P @ rho_true)))
        var = max(0.0, (1 - mu**2) / Ns)
        e = np.clip(mu + rng.normal(scale=np.sqrt(var)), -1.0, 1.0)
        rho_raw += e * P
    rho_raw /= d
    return project_density(rho_raw)

def reconstruct_for_theta(theta, n, Ns, p, seed, method):
    rho = depolarize(parameterized_ghz_density(theta), p)
    rho_ls = ls_qst_from_state(rho, n, Ns=Ns, seed=seed)
    if method == 'LS-QST':
        return rho_ls
    return ssp_qst(rho_ls, Ns=Ns)
