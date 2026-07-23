"""
SSP-QST: Spectral Subspace Purification for Photonic Quantum State Tomography.

Library layout:
  paulis        Pauli operators and the n-qubit measurement basis
  states        Probe-state construction (Haar rank-r mixtures, GHZ variants)
  simulator     Noise channels and the LS-QST measurement simulator
  purification  The SSP-QST method (Algorithm 1 of the paper)
  baselines     Non-iterative baselines (spectral squaring, top eigenvector)
  metrics       Fidelity and quantum Fisher information
  protocols     One deterministic experiment protocol per paper figure/table
"""
from .paulis import pauli_basis, kron_all, PAULIS, I2, X, Y, Z
from .states import (random_rank_state, ghz_state, physical_structured_ghz,
                     parameterized_ghz_density, project_density, density)
from .simulator import depolarize, ls_qst_from_state, reconstruct_for_theta
from .purification import ssp_qst
from .baselines import spectral_squaring, top_eigvec
from .metrics import fidelity, jz_operator, qfi
from .protocols import (compute_rank_fidelity, compute_gain_vs_noise,
                        compute_shot_efficiency, compute_rank_identification,
                        compute_qfi_recovery, compute_closed_loop,
                        compute_photonic_structured)

__version__ = "1.0.0"
