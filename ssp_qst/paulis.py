"""
paulis.py -- Pauli operators and informationally complete basis

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

Provides the single-qubit Pauli matrices, tensor products, and the
cached n-qubit Pauli basis {I, X, Y, Z}^{tensor n} used as the
informationally complete measurement set throughout the paper (Eq. 2).

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import itertools
import numpy as np

# Single-qubit Pauli matrices
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

# Cache: building the 4^n basis is O(4^n d^2); reuse across experiments.
pauli_cache = {}

def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def pauli_basis(n):
    if n not in pauli_cache:
        pauli_cache[n] = [kron_all(mats) for mats in itertools.product(PAULIS, repeat=n)]
    return pauli_cache[n]
