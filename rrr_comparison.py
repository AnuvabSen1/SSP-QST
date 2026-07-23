"""
rrr_comparison.py -- Runtime-matched comparison of SSP-QST against iterative
maximum-likelihood tomography (RrhoR / diluted MLE), at the exact Table II
configuration of the QCE26 paper (n=4, p=0.06, Ns=4096, seeds identical to
make_professional_figures.py compute_rank_fidelity).

RrhoR iteration (Rehacek-Hradil), Pauli two-outcome POVMs:
    Pi_{P,+-} = (I +- P)/2 for each non-identity Pauli P
    R(rho)    = sum_{P,s} f_{P,s} / p_{P,s}(rho) * Pi_{P,s}
    rho      <- N[ R rho R ]
Observed frequencies f_{P,+-} = (1 +- e_P)/2 built from the SAME noisy
expectations e_P used by the LS reconstruction (identical measurement record).

Two ML variants reported:
  RrhoR-RM   runtime-matched: the number of iterations whose wall time fits
             within the measured SSP-QST wall time (at least 1 granted).
  RrhoR-conv converged: iterations until ||rho_{k+1}-rho_k||_F < 1e-7
             (cap 400), wall time reported.

Deterministic; reproduces the numbers quoted in the paper's
"Runtime-Matched Comparison with Iterative Maximum Likelihood" subsection.
"""
import sys
import sys
import time
import numpy as np

# ---- functions copied verbatim from make_professional_figures.py ----------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS_1Q = [I2, X, Y, Z]

def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def pauli_basis(n):
    from itertools import product
    return [kron_all(c) for c in product(PAULIS_1Q, repeat=n)]

def project_density(rho):
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals, 0)
    s = vals.sum()
    if s <= 1e-15:
        vals = np.ones_like(vals) / len(vals)
    else:
        vals = vals / s
    return vecs @ np.diag(vals) @ vecs.conj().T

def random_rank_state(n, r, seed=0):
    rng = np.random.default_rng(seed)
    d = 2 ** n
    G = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(G)
    w = rng.dirichlet(np.ones(r))
    rho = (Q[:, :r] * w) @ Q[:, :r].conj().T
    return project_density(rho)

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

def fidelity(rho, sigma):
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals, 0)
    sq = vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T
    ev = np.linalg.eigvalsh(sq @ sigma @ sq)
    return float(np.sum(np.sqrt(np.maximum(ev, 0)))) ** 2

# ---- shared measurement record --------------------------------------------
def noisy_expectations(rho_true, n, Ns, seed):
    """Same noise draw as ls_qst_from_state (same rng consumption order)."""
    rng = np.random.default_rng(seed)
    es = []
    for P in pauli_basis(n):
        mu = float(np.real(np.trace(P @ rho_true)))
        var = max(0.0, (1 - mu**2) / Ns)
        es.append(float(np.clip(mu + rng.normal(scale=np.sqrt(var)), -1.0, 1.0)))
    return es

# ---- RrhoR iterative maximum likelihood -----------------------------------
def rrr_ml(exps, paulis, d, n_iter=None, tol=1e-7, cap=400):
    """RrhoR iterations from Pauli two-outcome frequencies.
    Returns (rho, iterations_used, seconds_per_iteration)."""
    fs, Ps = [], []
    for e, P in zip(exps, paulis):
        if np.allclose(P, np.eye(d)):
            continue
        f_plus = min(max((1.0 + e) / 2.0, 1e-12), 1.0 - 1e-12)
        fs.append((f_plus, 1.0 - f_plus))
        Ps.append(P)
    rho = np.eye(d, dtype=complex) / d
    Id = np.eye(d, dtype=complex)
    t0 = time.perf_counter()
    k = 0
    limit = n_iter if n_iter is not None else cap
    while k < limit:
        R = np.zeros((d, d), dtype=complex)
        for (fp, fm), P in zip(fs, Ps):
            pp = float(np.real(np.trace((Id + P) @ rho))) / 2.0
            pp = min(max(pp, 1e-12), 1.0 - 1e-12)
            R += (fp / pp) * (Id + P) / 2.0 + ((fm) / (1.0 - pp)) * (Id - P) / 2.0
        new = R @ rho @ R
        new = (new + new.conj().T) / 2
        new /= np.real(np.trace(new))
        delta = np.linalg.norm(new - rho)
        rho = new
        k += 1
        if n_iter is None and delta < tol:
            break
    secs = (time.perf_counter() - t0) / max(k, 1)
    return project_density(rho), k, secs

# ---- experiment ------------------------------------------------------------
def main():
    n, p, Ns, trials = 4, 0.06, 4096, 14
    d = 2 ** n
    paulis = pauli_basis(n)
    ranks = list(range(1, 8))

    # wall-time of SSP pipeline (LS post-processing only), median of repeats
    rho_demo = ls_qst_from_state(depolarize(random_rank_state(n, 3, seed=7), p), n, Ns, seed=11)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter(); ssp_qst(rho_demo, Ns=Ns); ts.append(time.perf_counter() - t0)
    t_ssp = float(np.median(ts))

    # single RrhoR iteration wall-time
    exps_demo = noisy_expectations(depolarize(random_rank_state(n, 3, seed=7), p), n, Ns, seed=11)
    _, _, t_iter = rrr_ml(exps_demo, paulis, d, n_iter=3)
    n_match = max(1, int(t_ssp // t_iter))

    print(f"SSP wall time (median): {t_ssp*1e3:.3f} ms")
    print(f"RrhoR per-iteration:    {t_iter*1e3:.3f} ms")
    print(f"Runtime-matched iters:  {n_match} (>=1 granted)")
    print()
    hdr = f"{'r':>2} | {'LS':>7} | {'SSP':>7} | {'RrR-RM':>7} | {'RrR-conv':>8} | {'iters':>5} | {'t_conv(s)':>9}"
    print(hdr); print("-" * len(hdr))

    rows = []
    for r in ranks:
        f_ls, f_ssp, f_rm, f_cv, its, tcv = [], [], [], [], [], []
        for t in range(trials):
            rho_tgt = random_rank_state(n, r, seed=1000 + 37*r + t)
            rho_noisy = depolarize(rho_tgt, p)
            seed_meas = 2000 + 19*r + t
            rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=seed_meas)
            exps = noisy_expectations(rho_noisy, n, Ns, seed=seed_meas)
            rho_ssp = ssp_qst(rho_ls, Ns=Ns)
            rho_rm, _, _ = rrr_ml(exps, paulis, d, n_iter=n_match)
            t0 = time.perf_counter()
            rho_cv, k_cv, _ = rrr_ml(exps, paulis, d, n_iter=None)
            tcv.append(time.perf_counter() - t0)
            f_ls.append(fidelity(rho_ls, rho_tgt))
            f_ssp.append(fidelity(rho_ssp, rho_tgt))
            f_rm.append(fidelity(rho_rm, rho_tgt))
            f_cv.append(fidelity(rho_cv, rho_tgt))
            its.append(k_cv)
        row = (r, np.mean(f_ls), np.mean(f_ssp), np.mean(f_rm), np.mean(f_cv),
               np.mean(its), np.mean(tcv))
        rows.append(row)
        print(f"{r:>2} | {row[1]:>7.4f} | {row[2]:>7.4f} | {row[3]:>7.4f} | "
              f"{row[4]:>8.4f} | {row[5]:>5.0f} | {row[6]:>9.3f}")

    with open("figs/rrr_comparison_summary.txt", "w") as f:
        f.write("Runtime-matched RrhoR comparison (n=4, p=0.06, Ns=4096, 14 trials)\n")
        f.write(f"SSP wall time {t_ssp*1e3:.3f} ms | RrhoR per-iter {t_iter*1e3:.3f} ms | RM iters {n_match}\n")
        for row in rows:
            f.write(f"r={row[0]}: LS={row[1]:.4f} SSP={row[2]:.4f} "
                    f"RrR-RM={row[3]:.4f} RrR-conv={row[4]:.4f} "
                    f"iters={row[5]:.0f} t_conv={row[6]:.3f}s\n")
    print("\nsaved figs/rrr_comparison_summary.txt")

if __name__ == "__main__" and "--diluted" not in sys.argv:
    main()


# ---- Diluted / step-optimised RrhoR (Rehacek et al. 2007) ------------------
def rrr_diluted(exps, paulis, d, tol=1e-7, cap=400,
                eps_grid=(0.5, 1.0, 2.0, 5.0)):
    """rho <- N[(I+eps R) rho (I+eps R)], eps chosen per step by
    log-likelihood line search (step-size-optimised diluted variant).
    Vectorised over the Pauli settings. Returns (rho, iters, seconds)."""
    Ps, fps = [], []
    for e, P in zip(exps, paulis):
        if np.allclose(P, np.eye(d)):
            continue
        Ps.append(P)
        fps.append(min(max((1.0 + e) / 2.0, 1e-12), 1.0 - 1e-12))
    P_stack = np.stack(Ps)                      # (m, d, d)
    fp = np.array(fps); fm = 1.0 - fp           # (m,)
    Id = np.eye(d, dtype=complex)

    def probs(rho):
        tr = np.real(np.einsum('pij,ji->p', P_stack, rho))
        return np.clip((1.0 + tr) / 2.0, 1e-12, 1.0 - 1e-12)

    def loglik(rho):
        pp = probs(rho)
        return float(np.sum(fp * np.log(pp) + fm * np.log(1.0 - pp)))

    rho = Id / d
    t0 = time.perf_counter()
    k = 0
    while k < cap:
        pp = probs(rho)
        w = fp / pp - fm / (1.0 - pp)           # (m,)
        R = (np.tensordot(w, P_stack, axes=1) / 2.0
             + np.sum(fp / pp + fm / (1.0 - pp)) * Id / 2.0)
        best, best_ll = None, -np.inf
        for eps in eps_grid:
            M = Id + eps * R
            cand = M @ rho @ M.conj().T
            cand = (cand + cand.conj().T) / 2
            cand /= np.real(np.trace(cand))
            ll = loglik(cand)
            if ll > best_ll:
                best_ll, best = ll, cand
        delta = np.linalg.norm(best - rho)
        rho = best
        k += 1
        if delta < tol:
            break
    return project_density(rho), k, time.perf_counter() - t0


def diluted_main():
    n, p, Ns, trials = 4, 0.06, 4096, 14
    d = 2 ** n
    paulis = pauli_basis(n)
    ranks = list(range(1, 8))
    print("Diluted RrhoR, per-step line search eps in {0.5,1,2,5}, "
          "tol 1e-7, cap 400 (identical seeds to main comparison)")
    hdr = f"{'r':>2} | {'SSP':>7} | {'Dil-RrR':>7} | {'iters':>5} | {'t(s)':>6}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for r in ranks:
        f_ssp, f_dil, its, ts = [], [], [], []
        for t in range(trials):
            rho_tgt = random_rank_state(n, r, seed=1000 + 37*r + t)
            rho_noisy = depolarize(rho_tgt, p)
            seed_meas = 2000 + 19*r + t
            rho_ls = ls_qst_from_state(rho_noisy, n, Ns=Ns, seed=seed_meas)
            exps = noisy_expectations(rho_noisy, n, Ns, seed=seed_meas)
            rho_ssp = ssp_qst(rho_ls, Ns=Ns)
            rho_d, k_d, t_d = rrr_diluted(exps, paulis, d)
            f_ssp.append(fidelity(rho_ssp, rho_tgt))
            f_dil.append(fidelity(rho_d, rho_tgt))
            its.append(k_d); ts.append(t_d)
        row = (r, np.mean(f_ssp), np.mean(f_dil), np.mean(its), np.mean(ts))
        rows.append(row)
        print(f"{r:>2} | {row[1]:>7.4f} | {row[2]:>7.4f} | {row[3]:>5.0f} | {row[4]:>6.3f}", flush=True)
    with open("figs/rrr_diluted_summary.txt", "w") as f:
        f.write("Diluted line-searched RrhoR (n=4, p=0.06, Ns=4096, 14 trials)\n")
        for row in rows:
            f.write(f"r={row[0]}: SSP={row[1]:.4f} DilRrR={row[2]:.4f} "
                    f"iters={row[3]:.0f} t={row[4]:.3f}s\n")
    print("saved figs/rrr_diluted_summary.txt")


if __name__ == "__main__" and "--diluted" in sys.argv:
    diluted_main()
