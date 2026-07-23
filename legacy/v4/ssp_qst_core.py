"""
ssp_qst_core.py — Spectral Subspace Purification QST (SSP-QST)
Georgia Tech GREEN Lab | Anuvab Sen | asen74@gatech.edu

Reviewer fixes applied:
  [R1] Physical parameter feedback: theta updated via fidelity gradient on
       parameterized ansatz circuit, NOT convex combination with target state.
  [R2] QFI is convex (not concave) in the state — corrected in comments.
  [R3] Renamed "MLE" baseline to "Spectral Squaring" (rho^2/tr(rho^2)).
  [R5] "Super-linear" replaced by "dimension-amplified" gain.
  [R6] Threshold derivation: p_hat/(2d) comes from 1-tr(rho^2) ≈ 2p(1-1/d)
       under depolarizing noise, so p/(2d) approximates noise eigenvalue floor.
  [R7] Noise models labeled "photonic-inspired", not "photonic-calibrated".
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error,
    amplitude_damping_error, phase_damping_error,
)
from qiskit.quantum_info import DensityMatrix

# ─────────────────────────────────────────────────────────────────
# Photonic-inspired noise models  [FIX R7]
# ─────────────────────────────────────────────────────────────────

def photonic_noise_model(n_qubits, p_dep=0., p_amp=0., p_phase=0.):
    """
    Qiskit Aer NoiseModel with photonic-inspired gate-level errors.

    Physical motivation (not claimed as experimentally calibrated):
      p_dep   : mode mismatch / coupling inefficiency at beam splitters
      p_amp   : amplitude damping modelling photon loss/absorption
      p_phase : phase damping modelling path-length / dephasing noise

    Note on amplitude damping: Qiskit's amplitude_damping_error maps
    |1> -> sqrt(1-gamma)|1>, |0> + sqrt(gamma)|0> — a reasonable
    approximation for single-photon loss into a vacuum reservoir.
    True photonic erasure (loss to unmeasured mode) is more complex;
    we use this as an idealized proxy consistent with prior literature.
    """
    nm = NoiseModel()
    gates_1q = ['h', 'rx', 'ry', 'rz', 'u', 'sdg']
    gates_2q = ['cx', 'cz']

    if p_dep > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p_dep, 1), gates_1q)
        # 2-qubit gates typically have ~2× noise floor
        nm.add_all_qubit_quantum_error(depolarizing_error(min(p_dep * 2, 1.), 2), gates_2q)

    if p_amp > 0:
        nm.add_all_qubit_quantum_error(amplitude_damping_error(p_amp), gates_1q)

    if p_phase > 0:
        # phase_damping_error models T2 dephasing (not phase-flip)
        nm.add_all_qubit_quantum_error(phase_damping_error(p_phase), gates_1q)

    return nm


# ─────────────────────────────────────────────────────────────────
# State preparation circuits
# ─────────────────────────────────────────────────────────────────

def ghz_circuit(n):
    """n-qubit GHZ: H(q0), CX(q0,q1), ..., CX(q_{n-2},q_{n-1})."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def bell_circuit():
    """2-qubit Bell state Phi+ = (|00>+|11>)/sqrt(2)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def parameterized_ghz_ansatz(n, theta):
    """
    Parameterized ansatz for physical feedback.  [FIX R1]

    Starts from the GHZ circuit and adds a layer of single-qubit
    Ry(theta_i) rotations — simulating the effect of wave-plate angle
    adjustments (theta_i) on the photonic source.

    theta : 1-D array of length n (one angle per qubit).
    This represents the physically controllable parameters of the source
    (e.g., HWP angles, EOM voltages) that the feedback controller adjusts.
    """
    assert len(theta) == n
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    # Physical perturbation layer: small rotations around current operating point
    for i in range(n):
        qc.ry(float(theta[i]), i)
    return qc


# ─────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────

_SIM_DM = AerSimulator(method='density_matrix')


def exact_density_matrix(qc, noise_model=None):
    """Exact density matrix from Qiskit Aer density_matrix simulator."""
    sim = AerSimulator(method='density_matrix',
                       noise_model=noise_model) if noise_model else _SIM_DM
    qc2 = qc.copy()
    qc2.save_density_matrix()
    res = sim.run(transpile(qc2, sim), shots=1).result()
    return np.array(res.data()['density_matrix'].data, dtype=complex)


# ─────────────────────────────────────────────────────────────────
# Pauli tools
# ─────────────────────────────────────────────────────────────────

_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1., -1.]).astype(complex),
}


def pauli_matrix(s):
    """Tensor-product Pauli matrix from string e.g. 'XYZ'."""
    m = _PAULI[s[0]]
    for c in s[1:]:
        m = np.kron(m, _PAULI[c])
    return m


def pauli_strings(n):
    """All 4^n - 1 non-identity n-qubit Pauli strings."""
    from itertools import product
    return [''.join(c) for c in product('IXYZ', repeat=n)
            if not all(x == 'I' for x in c)]


def sample_expectations(rho, n, shots, rng):
    """
    Add shot noise to exact Pauli expectations.
    Var(e_k) = (1 - e_k^2) / shots  [Bernoulli variance for 2-outcome mmt]
    """
    exps = {}
    for ps in pauli_strings(n):
        P = pauli_matrix(ps)
        e = float(np.real(np.trace(P @ rho)))
        sigma = np.sqrt(max(0., 1. - e**2) / shots)
        exps[ps] = float(np.clip(e + rng.normal(0., sigma), -1., 1.))
    return exps


# ─────────────────────────────────────────────────────────────────
# Reconstruction methods
# ─────────────────────────────────────────────────────────────────

def project_physical(rho):
    """Project onto valid density matrix (PSD + trace-1)."""
    rho_h = (rho + rho.conj().T) / 2.
    vals, vecs = eigh(rho_h)
    vals = np.maximum(vals, 0.)
    s = vals.sum()
    if s < 1e-15:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    vals /= s
    return (vecs * vals) @ vecs.conj().T


def ls_reconstruct(exps, n):
    """
    Least-squares QST: rho_LS = Pi_phys[I/d + (1/d) sum_k e_k P_k].
    """
    d = 2 ** n
    rho = np.eye(d, dtype=complex) / d
    for ps, ek in exps.items():
        rho += (ek / d) * pauli_matrix(ps)
    return project_physical(rho)


def spectral_squaring(rho_ls):
    """
    Spectral squaring baseline: rho^2 / tr(rho^2).  [FIX R3]
    Renamed from "one-step MLE" — this is NOT iterative MLE.
    It is a purity-boosting heuristic that sharpens the eigenspectrum
    by squaring eigenvalues, not a likelihood-maximizing algorithm.
    """
    rho_sq = rho_ls @ rho_ls
    tr = float(np.real(np.trace(rho_sq)))
    return project_physical(rho_sq / max(tr, 1e-12))


def ssp_purify(rho_ls, shots):
    """
    Spectral Subspace Purification QST (SSP-QST).

    Threshold derivation  [FIX R6]:
      Under depolarizing noise p:  1 - tr(rho^2) ≈ 2p(1 - 1/d)
      => p ≈ p_hat / (2*(1 - 1/d)) ≈ p_hat / 2  for large d
      Noise eigenvalues scale as p/d ≈ p_hat / (2d).
      Shot-noise adds O(1/sqrt(shots)) to each eigenvalue.
      => threshold = p_hat/(2d) + 0.5/sqrt(shots)

    This is a Weyl-perturbation-motivated heuristic, not a strict bound.
    For non-depolarizing noise the same formula provides a conservative
    noise floor estimate because 1 - tr(rho^2) over-estimates mixing.
    """
    d = rho_ls.shape[0]
    rho_h = (rho_ls + rho_ls.conj().T) / 2.
    vals, vecs = eigh(rho_h)          # ascending order, real eigenvalues
    vals = np.maximum(vals, 0.)

    # Refined threshold derivation  [Reviewer round 2]:
    # For depolarizing noise p:
    #   lambda_max = 1 - p(d-1)/d  =>  p_hat = 1-lambda_max = p(d-1)/d
    #   noise eigenvalues = p/d = p_hat/(d-1)
    # So the correct noise-floor scale is p_hat/(d-1), not p_hat/d.
    # For large d the difference is small, but the derivation is cleaner.
    # Shot noise adds O(0.5/sqrt(N_s)) per eigenvalue.
    lambda_max = vals[-1]   # largest (eigh gives ascending order)
    p_hat = max(0., 1. - lambda_max)
    eps = p_hat / max(d - 1, 1) + 0.5 / np.sqrt(max(shots, 1))

    vals_p = np.where(vals > eps, vals, 0.)
    if vals_p.sum() < 1e-12:
        vals_p[-1] = 1.          # fallback: keep dominant eigenvector
    vals_p /= vals_p.sum()

    return (vecs * vals_p) @ vecs.conj().T


# ─────────────────────────────────────────────────────────────────
# Physical feedback controller  [FIX R1]
# ─────────────────────────────────────────────────────────────────

def fidelity_np(rho, sigma):
    """F(rho, sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    vals, vecs = eigh(rho)
    vals = np.maximum(vals, 0.)
    sqrt_rho = (vecs * np.sqrt(vals)) @ vecs.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    ev = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(ev, 0.)))) ** 2


def physical_feedback_step(theta, n, noise_model, rho_target,
                            rho_ssp, eta=0.15, rng=None, shots=4096):
    """
    Physical parameter feedback.  [FIX R1]

    Updates wave-plate angles theta via fidelity gradient:
        theta_{t+1} = theta_t + eta * grad_theta F(rho(theta_t), rho_target)

    Gradient estimated by parameter-shift rule:
        dF/d_theta_i ≈ [F(theta + pi/2 e_i) - F(theta - pi/2 e_i)] / 2

    The gradient is computed on the SSP-QST estimate to reduce noise,
    not on the raw LS reconstruction — this is the key advantage of
    embedding SSP inside the feedback loop.

    Returns: updated theta, new noisy density matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    grad = np.zeros(n)
    for i in range(n):
        def _eval(delta):
            th = theta.copy()
            th[i] += delta
            qc = parameterized_ghz_ansatz(n, th)
            rho_n = exact_density_matrix(qc, noise_model)
            # Add shot noise
            exps = sample_expectations(rho_n, n, shots, rng)
            rho_ls = ls_reconstruct(exps, n)
            rho_ssp_i = ssp_purify(rho_ls, shots)
            return fidelity_np(rho_ssp_i, rho_target)

        grad[i] = (_eval(np.pi / 2.) - _eval(-np.pi / 2.)) / 2.

    theta_new = theta + eta * grad

    # Evaluate at new theta
    qc_new = parameterized_ghz_ansatz(n, theta_new)
    rho_new_noisy = exact_density_matrix(qc_new, noise_model)
    exps_new = sample_expectations(rho_new_noisy, n, shots, rng)
    rho_ls_new = ls_reconstruct(exps_new, n)
    rho_ssp_new = ssp_purify(rho_ls_new, shots)

    return theta_new, rho_ssp_new, fidelity_np(rho_ssp_new, rho_target)


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def jz(n):
    """J_z = 0.5 * sum_i Z_i.  F_Q_max = n^2 for GHZ (Heisenberg limit)."""
    Z = np.diag([1., -1.]).astype(complex)
    I2 = np.eye(2, dtype=complex)
    H = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n):
        op = np.eye(1, dtype=complex)
        for r in range(n):
            op = np.kron(op, Z if r == q else I2)
        H += op
    return 0.5 * H


def qfi(rho, H):
    """
    QFI: F_Q[rho, H] = 2 sum_{i!=j} (l_i-l_j)^2/(l_i+l_j) |<i|H|j>|^2

    QFI is CONVEX in rho for fixed H (not concave).  [FIX R2]
    Mixing states (noise) can only reduce QFI:
        F_Q[p*rho + (1-p)*sigma, H] <= p*F_Q[rho,H] + (1-p)*F_Q[sigma,H]
    is WRONG — in fact F_Q is convex so mixing REDUCES it.
    The correct statement: QFI(mixed state) <= QFI(pure state component).
    """
    vals, vecs = eigh(rho)
    vals = np.maximum(vals, 0.)
    H_eb = vecs.conj().T @ H @ vecs
    result = 0.
    d = len(vals)
    for i in range(d):
        for j in range(d):
            s = vals[i] + vals[j]
            if s > 1e-14:
                result += 2. * (vals[i] - vals[j])**2 / s * abs(H_eb[i, j])**2
    return float(result)


# ─────────────────────────────────────────────────────────────────
# Mixed-rank target states  (for Reviewer experiment R-mix)
# ─────────────────────────────────────────────────────────────────

def w_state(n):
    """
    n-qubit W state |W_n> = (1/sqrt(n)) sum_i |0..010..0>_i
    W state is rank-1 (pure) but has a very different entanglement
    structure from GHZ. Used as the second component of the mixed target.
    """
    d = 2**n
    psi = np.zeros(d, dtype=complex)
    for i in range(n):
        # index with a single 1 at position i (big-endian)
        idx = 1 << (n - 1 - i)
        psi[idx] = 1.0 / np.sqrt(n)
    return np.outer(psi, psi.conj())


def ghz_density(n):
    """GHZ density matrix as numpy array (no Qiskit needed)."""
    d = 2**n
    psi = np.zeros(d, dtype=complex)
    psi[0]  = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj())


def mixed_rank2_target(n, alpha=0.65):
    """
    Rank-2 mixed target: alpha * |GHZ><GHZ| + (1-alpha) * |W><W|
    This is a physically motivated probe: a probabilistic mixture of two
    entangled photonic states (e.g., from an imperfect GHZ source that
    occasionally produces the W state due to a misaligned BS).
    Rank = 2 by construction; all rank-1 methods (spectral squaring,
    top-eigenvector extraction) are suboptimal here.
    alpha: mixing weight. Default 0.65 gives a well-conditioned rank-2 state.
    """
    rho_ghz = ghz_density(n)
    rho_w   = w_state(n)
    rho_mix = alpha * rho_ghz + (1.0 - alpha) * rho_w
    # Check physical validity
    vals = np.linalg.eigvalsh(rho_mix)
    assert np.all(vals >= -1e-10), "Mixed state not PSD"
    assert abs(np.trace(rho_mix) - 1.0) < 1e-10, "Trace not 1"
    return rho_mix
