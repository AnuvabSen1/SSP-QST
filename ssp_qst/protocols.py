"""
protocols.py -- Experiment protocols (one per paper figure/table)

Part of the SSP-QST codebase for the IEEE QCE 2026 paper
"SSP-QST: Spectral Subspace Purification for Photonic Quantum State
Tomography" (Paper ID QPHO-981).

Each compute_* function runs one experiment of the paper and returns
plain arrays; all seeds are fixed inside, so every call is deterministic:
- compute_rank_fidelity      -> Fig. 3, Fig. 4a, Table II
- compute_gain_vs_noise      -> Fig. 4b
- compute_shot_efficiency    -> Fig. 6a, Table III
- compute_rank_identification-> Fig. 6b
- compute_qfi_recovery       -> Fig. 5a
- compute_closed_loop        -> Fig. 5c
- compute_photonic_structured-> Fig. 5b

All numerical code is verbatim from the verified reference implementation
that reproduces the paper byte-for-byte; only documentation is added.
"""

import numpy as np

from .states import (random_rank_state, physical_structured_ghz,
                     parameterized_ghz_density, project_density)
from .simulator import depolarize, ls_qst_from_state, reconstruct_for_theta
from .purification import ssp_qst
from .baselines import spectral_squaring, top_eigvec
from .metrics import fidelity, jz_operator, qfi

def compute_rank_fidelity(n=4, p=0.06, Ns=4096, trials=14):
    ranks = list(range(1, 8))
    methods = ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']
    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}
    for r in ranks:
        vals = {m: [] for m in methods}
        for t in range(trials):
            rho_tgt = random_rank_state(n, r, seed=1000 + 37*r + t)
            rho_noisy = depolarize(rho_tgt, p)
            rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=2000 + 19*r + t)
            recs = {
                'LS-QST': rho_ls,
                'Spec. Sq.': spectral_squaring(rho_ls),
                'Top Eigvec.': top_eigvec(rho_ls),
                'SSP-QST': ssp_qst(rho_ls, Ns=Ns),
            }
            for m in methods:
                vals[m].append(fidelity(recs[m], rho_tgt))
        for m in methods:
            means[m].append(float(np.mean(vals[m])))
            stds[m].append(float(np.std(vals[m], ddof=1)))
    return ranks, means, stds

def compute_gain_vs_noise(n=4, ranks=(3,5), p_vals=None, Ns=4096, trials=12):
    if p_vals is None:
        p_vals = np.linspace(0.0, 0.20, 6)
    out = {r: [] for r in ranks}
    for r in ranks:
        for p in p_vals:
            gains = []
            for t in range(trials):
                rho_tgt = random_rank_state(n, r, seed=4000 + r*100 + t)
                rho_noisy = depolarize(rho_tgt, float(p))
                rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=5000 + r*100 + t)
                f_ssp = fidelity(ssp_qst(rho_ls, Ns=Ns), rho_tgt)
                baselines = [
                    fidelity(rho_ls, rho_tgt),
                    fidelity(spectral_squaring(rho_ls), rho_tgt),
                    fidelity(top_eigvec(rho_ls), rho_tgt),
                ]
                gains.append(f_ssp - max(baselines))
            out[r].append(float(np.mean(gains)))
    return np.array(p_vals), out

def compute_shot_efficiency(n=4, r=3, p=0.08, Nss=(512,1024,2048,4096), trials=12):
    methods = ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']
    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}
    for Ns in Nss:
        vals = {m: [] for m in methods}
        for t in range(trials):
            rho_tgt = random_rank_state(n, r, seed=7000 + t)
            rho_noisy = depolarize(rho_tgt, p)
            rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=8000 + t)
            recs = {
                'LS-QST': rho_ls,
                'Spec. Sq.': spectral_squaring(rho_ls),
                'Top Eigvec.': top_eigvec(rho_ls),
                'SSP-QST': ssp_qst(rho_ls, Ns=Ns),
            }
            for m in methods:
                vals[m].append(fidelity(recs[m], rho_tgt))
        for m in methods:
            means[m].append(float(np.mean(vals[m])))
            stds[m].append(float(np.std(vals[m], ddof=1)))
    return list(Nss), means, stds

def compute_rank_identification(n=4, true_ranks=range(2,7), p_vals=(0.02,0.05,0.08,0.11,0.14), Ns=4096, trials=20):
    mat = np.zeros((len(p_vals), len(list(true_ranks))))
    for pi, p in enumerate(p_vals):
        for ri, r in enumerate(true_ranks):
            ranks = []
            for t in range(trials):
                rho_tgt = random_rank_state(n, r, seed=9000 + 100*pi + 10*ri + t)
                rho_noisy = depolarize(rho_tgt, p)
                rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=10000 + 100*pi + 10*ri + t)
                _, rr = ssp_qst(rho_ls, Ns=Ns, return_rank=True)
                ranks.append(rr)
            mat[pi, ri] = np.mean(ranks)
    return list(true_ranks), list(p_vals), mat

def compute_qfi_recovery(n=4, p=0.06, Ns=4096, ranks=range(2,8), trials=12):
    methods = ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']
    means = {m: [] for m in methods}
    H = jz_operator(n)
    for r in ranks:
        vals = {m: [] for m in methods}
        for t in range(trials):
            rho_tgt = random_rank_state(n, r, seed=11000 + 10*r + t)
            rho_noisy = depolarize(rho_tgt, p)
            rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=12000 + 10*r + t)
            recs = {
                'LS-QST': rho_ls,
                'Spec. Sq.': spectral_squaring(rho_ls),
                'Top Eigvec.': top_eigvec(rho_ls),
                'SSP-QST': ssp_qst(rho_ls, Ns=Ns),
            }
            target_qfi = max(qfi(rho_tgt, H), 1e-12)
            for m in methods:
                q_est = qfi(recs[m], H)
                vals[m].append(max(0.0, 1.0 - abs(q_est - target_qfi) / target_qfi))
        for m in methods:
            means[m].append(float(np.mean(vals[m])))
    return list(ranks), means

def compute_closed_loop(n=3, Ns=2048, p=0.06, steps=20, eta=0.35):
    methods = ['LS-QST', 'SSP-QST']
    target = parameterized_ghz_density(np.zeros(n))
    histories = {}
    for method in methods:
        theta = np.array([0.38, -0.28, 0.31])[:n].astype(float)
        hist = []
        for it in range(steps):
            rho_rec = reconstruct_for_theta(theta, n, Ns, p, seed=13000 + it, method=method)
            hist.append(fidelity(rho_rec, target))
            grad = np.zeros(n)
            for i in range(n):
                e = np.zeros(n); e[i] = 1.0
                rho_p = reconstruct_for_theta(theta + np.pi/2 * e, n, Ns, p, seed=14000 + 10*it + i, method=method)
                rho_m = reconstruct_for_theta(theta - np.pi/2 * e, n, Ns, p, seed=15000 + 10*it + i, method=method)
                grad[i] = 0.5 * (fidelity(rho_p, target) - fidelity(rho_m, target))
            theta = theta + eta * grad
        histories[method] = hist
    return histories

def compute_photonic_structured(n=4, Ns=4096, p=0.04, crosstalks=(0.0,0.02,0.04,0.06,0.08), trials=12):
    methods = ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']
    means = {m: [] for m in methods}
    for c in crosstalks:
        vals = {m: [] for m in methods}
        for t in range(trials):
            rho_tgt = physical_structured_ghz(n, leakage=0.05, crosstalk=float(c), dephase=0.10)
            rho_noisy = depolarize(rho_tgt, p)
            rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=16000 + 100*int(c*100) + t)
            recs = {
                'LS-QST': rho_ls,
                'Spec. Sq.': spectral_squaring(rho_ls),
                'Top Eigvec.': top_eigvec(rho_ls),
                'SSP-QST': ssp_qst(rho_ls, Ns=Ns),
            }
            for m in methods:
                vals[m].append(fidelity(recs[m], rho_tgt))
        for m in methods:
            means[m].append(float(np.mean(vals[m])))
    return list(crosstalks), means

# ------------------
# Plotting
# ------------------
