"""
states.py -- Probe-state construction

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

Target photonic probe states used in the experiments:
- random_rank_state: Haar-frame rank-r mixtures (Ginibre + QR gives a
  Haar-distributed orthonormal r-frame; weights are flat Dirichlet), the
  sampling procedure specified in Sec. Simulation Setup of the paper.
- ghz_state / physical_structured_ghz: GHZ probes, ideal and with
  leakage, dephasing, and crosstalk (Fig. 5b).
- parameterized_ghz_density: wave-plate-angle parameterised GHZ source
  for the closed-loop drift-correction experiment (Fig. 5c).

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import numpy as np

from .paulis import kron_all

def project_density(rho):
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals, 0)
    if vals.sum() <= 1e-15:
        vals[-1] = 1.0
    vals = vals / vals.sum()
    return vecs @ np.diag(vals) @ vecs.conj().T

def density(psi):
    return np.outer(psi, psi.conj())

def ghz_state(n):
    d = 2 ** n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = 1/np.sqrt(2)
    return psi

def random_rank_state(n, r, seed=0):
    rng = np.random.default_rng(seed)
    d = 2 ** n
    A = rng.normal(size=(d, r)) + 1j * rng.normal(size=(d, r))
    Q, _ = np.linalg.qr(A)
    w = rng.dirichlet(np.ones(r))
    rho = np.zeros((d, d), dtype=complex)
    for i in range(r):
        rho += w[i] * np.outer(Q[:, i], Q[:, i].conj())
    return project_density(rho)

def physical_structured_ghz(n, leakage=0.05, crosstalk=0.04, dephase=0.10):
    d = 2 ** n
    rho = density(ghz_state(n))
    rho = rho.copy()
    rho[0, -1] *= (1 - dephase)
    rho[-1, 0] *= (1 - dephase)

    ket0 = np.zeros(d, dtype=complex); ket0[0] = 1
    ket1 = np.zeros(d, dtype=complex); ket1[-1] = 1
    rho_leak = 0.5 * density(ket0) + 0.5 * density(ket1)

    rho_cross = np.zeros((d, d), dtype=complex)
    for q in range(n):
        a = 1 << q
        b = (d - 1) ^ (1 << q)
        phi = np.zeros(d, dtype=complex)
        phi[a] = 1/np.sqrt(2)
        phi[b] = 1/np.sqrt(2)
        rho_cross += density(phi)
    rho_cross /= n

    rho_mix = (1 - leakage - crosstalk) * rho + leakage * rho_leak + crosstalk * rho_cross
    return project_density(rho_mix)

def ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def parameterized_ghz_density(theta):
    U = kron_all([ry(th) for th in theta])
    psi = U @ ghz_state(len(theta))
    return density(psi)
