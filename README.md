# SSP-QST Codebase — QCE26 Paper QPHO-981
## SSP-QST: Spectral Subspace Purification for Photonic Quantum State Tomography
### Anuvab Sen, Saibal Mukhopadhyay | Georgia Tech | asen74@gatech.edu

## AUTHORITATIVE SCRIPT (produces all camera-ready figures and tables)
    pip install numpy scipy matplotlib
    python make_professional_figures.py

Runtime approximately 10-20 minutes. Outputs to ./figs/ (paths are
script-relative). Fonts are set to 11pt axes / 10pt ticks-legend per the
camera-ready (Reviewer 2 request); figure data is unaffected by font
settings and reproduces the results summary byte-for-byte. All randomness is
deterministically seeded; re-running reproduces figs/selected_results_summary.txt
byte-for-byte, which matches the camera-ready paper:
- Table II fidelity vs rank (n=4, p=0.06, Ns=4096), all rows
- Max gain +0.584 over top-eigenvector at rank 7 (0.9412 - 0.3576)
- Max gain +0.051 over spectral squaring at rank 4 (0.9874 - 0.9368)
- >= 8x photon budget: SSP at Ns=512 (0.956) beats LS at Ns=4096 (0.890)
- Table IV threshold ablation reproduced from a single rank-3 reference target
  (default and p_hat/d: rank 3, F = 0.997; p_hat/(2d): rank 4, F ~ 0.97)


## rrr_comparison.py (runtime-matched iterative ML comparison)
    python rrr_comparison.py
Runs the RrhoR (Rehacek-Hradil) maximum-likelihood iteration on the identical
measurement record, seeds, and targets as Table II (n=4, p=0.06, Ns=4096).
Reproduces the paper's "Runtime-Matched Comparison with Iterative Maximum
Likelihood" subsection: SSP wall time 0.16 ms vs 4.0 ms per RrhoR iteration
(runtime-matched budget admits at most 1 iteration, fidelity 0.07-0.37);
400 iterations (~1.5 s, ~1e4x compute) reach mean fidelity 0.90-0.94,
below SSP-QST at every rank and below projected LS for ranks >= 4.
Output: figs/rrr_comparison_summary.txt

## rrr_comparison.py (runtime-matched iterative-ML comparison)
    python rrr_comparison.py            # undiluted RrhoR, two budgets
    python rrr_comparison.py --diluted  # step-size-optimised diluted variant
Reproduces the paper's "Runtime-Matched Comparison with Iterative Maximum
Likelihood" subsection at the exact Table II configuration (n=4, p=0.06,
Ns=4096, identical seeds). Verified output: SSP median wall time ~0.16 ms,
one RrhoR iteration ~4 ms; runtime-matched budget admits 1 iteration
(fidelity 0.07-0.37 across ranks); converged 400-iteration RrhoR reaches
0.936, 0.923, 0.908, 0.900, 0.897, 0.897, 0.899 at ranks 1-7, trailing
SSP-QST at every rank. The --diluted mode runs the line-searched diluted
iteration (eps in {0.5,1,2,5} per step): it converges to the same fixed
point as undiluted RrhoR (mean fidelities within 1e-3 at every rank,
figs/rrr_diluted_summary.txt), confirming the gap to SSP-QST is a
property of the unregularised ML estimate, not incomplete convergence.
Wall times are machine-dependent; fidelities are deterministic.

## verify_bounds.py (theoretical-bound verification)
    python verify_bounds.py
Reproduces the two bound checks quoted in the paper's threshold-scope
discussion: (1) the amplitude-damping envelope, elementary diamond-norm
bound gamma + 2a + a^2 <= 2 gamma (1+gamma) with a = 1 - sqrt(1-gamma)
(exact identity 2a - a^2 = gamma; numeric diamond norm = 2 gamma), full
n-qubit shifts observed within n gamma; (2) the matrix-Bernstein
envelope sqrt(2 ln(2d)/Ns) = 0.041 and the margin ablation (rank-7
fidelity 0.941 -> 0.786 if the worst case replaces 0.5/sqrt(Ns)).

## legacy/ (earlier development versions, included for provenance)
- legacy/v4/ssp_qst_core.py    Qiskit Aer pipeline: GHZ circuits, photonic-
                               inspired noise models (depolarizing, amplitude
                               damping, phase damping), LS-QST, SSP purification,
                               parameter-shift feedback controller.
                               Requires: pip install qiskit qiskit-aer
                               (verified working on Qiskit 2.5.1 + Aer 0.17.2)
- legacy/v5/rank_experiments.py  n=3 rank-sweep experiments built on the v4 core.
                               Edit the sys.path.insert line to point at the
                               directory containing ssp_qst_core.py before running.

## Reconciliation note (legacy vs camera-ready)
The v4/v5 legacy code uses the SAME noise-floor threshold as the paper,
eps = p_hat/(d-1) + 0.5/sqrt(Ns), but PRE-DATES the rank-1 override step
(Algorithm 1, line 6: if lambda_{d-1} < 2*eps return top eigenvector).
Consequently legacy rank-1 fidelities (~0.95) are lower than the camera-ready
Table II rank-1 value (1.000); adding the override closes exactly that gap.
Legacy experiments are also n=3 at p=0.08, while all camera-ready results are
n=4 (make_professional_figures.py). The legacy runs still show the paper's
qualitative claims: SSP-QST wins at every mixed rank r=2,3,4 where rank-1
purification collapses (e.g. n=3: r=2 SSP 0.994 vs TopEig 0.635, r=3 SSP 0.984
vs 0.473, r=4 SSP 0.968 vs 0.451).

## Key functions (make_professional_figures.py)
- random_rank_state(n, r, seed)   Haar frame (Ginibre + QR) x Dirichlet(1,...,1)
- ls_qst_from_state(...)          LS-QST, analytic Bernoulli shot noise (Eq. 3-4)
- ssp_qst(rho_ls, Ns)             Algorithm 1: eigh, Weyl noise floor, rank-1
                                  override, threshold, renormalise
- spectral_squaring, top_eigvec   Non-iterative baselines
- qfi(rho, H)                     Quantum Fisher information (Eq. 5)
- physical_structured_ghz(...)    Leakage/dephasing/crosstalk GHZ (Fig. 6b)
- compute_closed_loop(...)        Parameter-shift feedback (Fig. 6c)

## Note on figE
figE_additional_validation_improved.pdf in the paper is a styling pass over
figE_additional_validation from this script; data and panels are identical.
