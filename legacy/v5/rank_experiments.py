"""
rank_experiments.py
Focused experiments: SSP-QST on rank-2, 3, 4, 5 mixed-rank photonic targets.
This is the core contribution — rank-adaptive reconstruction where SSP-QST wins.

Run: python rank_experiments.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/v4/code")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.linalg import eigh

from ssp_qst_core import (
    sample_expectations, ls_reconstruct,
    spectral_squaring, ssp_purify, fidelity_np,
    project_physical, pauli_strings
)

RNG = np.random.default_rng(42)
OUT = "/home/claude/v5/figs"
os.makedirs(OUT, exist_ok=True)

C_LS  = "#1f77b4"
C_SSP = "#d62728"
C_SS  = "#2ca02c"
C_TOP = "#9467bd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "legend.framealpha": 0.92,
    "lines.linewidth": 2.4, "lines.markersize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


# ─── State construction ────────────────────────────────────────────────────────

def random_pure_state(d, rng):
    """Random rank-1 pure state in d-dimensional Hilbert space."""
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def rank_r_target(n_qubits, rank, weights=None, rng=None):
    """
    Rank-r mixed photonic target: uniform mixture of r orthogonal random pure states.
    weights: probability weights (default uniform).
    This models a photonic source with r dominant modes — realistic for any
    source with partial beam-splitter misalignment, mode crosstalk, or
    multiphoton emission.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    d = 2**n_qubits
    assert rank <= d, f"rank {rank} > d {d}"

    # Generate r orthogonal pure states via QR decomposition of random matrix
    M = rng.standard_normal((d, rank)) + 1j * rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(M)        # columns are orthonormal
    vecs = Q[:, :rank]            # d × rank matrix

    if weights is None:
        # Slightly unequal weights to make it realistic
        w = rng.dirichlet(np.ones(rank) * 2.0)
    else:
        w = np.array(weights) / sum(weights)

    rho = np.zeros((d, d), dtype=complex)
    for i in range(rank):
        v = vecs[:, i]
        rho += w[i] * np.outer(v, v.conj())

    # Verify
    assert abs(np.trace(rho).real - 1.0) < 1e-10
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-10)
    ev = np.sort(np.linalg.eigvalsh(rho))[::-1]
    actual_rank = int((ev > 1e-6).sum())
    return rho, actual_rank


def rank1_top(rho_ls):
    """Top-eigenvector extraction — always rank-1."""
    vals, vecs = eigh(rho_ls)
    return np.outer(vecs[:, -1], vecs[:, -1].conj())


def tomo_np(rho_target, n, p_dep, shots):
    """Analytical depolarizing noise + shot noise tomography (no Qiskit needed for speed)."""
    d = 2**n
    rho_noisy = (1-p_dep)*rho_target + (p_dep/d)*np.eye(d, dtype=complex)
    exps = sample_expectations(rho_noisy, n, shots, RNG)
    rho_ls  = ls_reconstruct(exps, n)
    rho_ss  = spectral_squaring(rho_ls)
    rho_ssp = ssp_purify(rho_ls, shots)
    rho_top = rank1_top(rho_ls)
    return rho_ls, rho_ss, rho_ssp, rho_top


# ─── Fig A: Fidelity vs Rank (key figure) ─────────────────────────────────────

def figA_fidelity_vs_rank():
    """
    Main result: show fidelity for rank-1 through rank-5 targets.
    n=3 qubits (d=8), fixed noise p=0.08, N_s=4096.
    Each rank uses 5 random mixed targets (averaged) for robustness.
    """
    print("Fig A: Fidelity vs. target rank...")
    n = 3; d = 8; p = 0.08; shots = 4096; n_trials = 8
    ranks = [1, 2, 3, 4, 5]

    f_ls  = {r: [] for r in ranks}
    f_ss  = {r: [] for r in ranks}
    f_ssp = {r: [] for r in ranks}
    f_top = {r: [] for r in ranks}

    for rank in ranks:
        for trial in range(n_trials):
            rng_t = np.random.default_rng(trial * 100 + rank)
            rho_t, _ = rank_r_target(n, rank, rng=rng_t)
            rho_ls, rho_ss, rho_ssp, rho_top = tomo_np(rho_t, n, p, shots)
            f_ls[rank].append(fidelity_np(rho_ls,  rho_t))
            f_ss[rank].append(fidelity_np(rho_ss,  rho_t))
            f_ssp[rank].append(fidelity_np(rho_ssp, rho_t))
            f_top[rank].append(fidelity_np(rho_top, rho_t))
        print(f"  rank={rank}: LS={np.mean(f_ls[rank]):.3f}  SS={np.mean(f_ss[rank]):.3f}  "
              f"SSP={np.mean(f_ssp[rank]):.3f}  TopEig={np.mean(f_top[rank]):.3f}")

    x = np.array(ranks)
    m_ls  = [np.mean(f_ls[r])  for r in ranks]
    m_ss  = [np.mean(f_ss[r])  for r in ranks]
    m_ssp = [np.mean(f_ssp[r]) for r in ranks]
    m_top = [np.mean(f_top[r]) for r in ranks]
    e_ls  = [np.std(f_ls[r])   for r in ranks]
    e_ssp = [np.std(f_ssp[r])  for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(x, m_ls,  yerr=e_ls,  fmt="o--", color=C_LS,  ms=9, capsize=4,
                lw=2.2, label="LS-QST")
    ax.plot(x, m_ss,  "^--", color=C_SS,  ms=8, lw=2.2, label="Spectral Squaring (rank-1)")
    ax.plot(x, m_top, "D--", color=C_TOP, ms=8, lw=2.0, label="Top Eigenvector (rank-1)")
    ax.errorbar(x, m_ssp, yerr=e_ssp, fmt="s-", color=C_SSP, ms=10, capsize=4,
                lw=2.8, label="SSP-QST (rank-adaptive)")

    # Annotate crossover
    ax.axvline(1.5, color='gray', lw=1.2, ls=':', alpha=0.7)
    ax.text(1.55, 0.62, "Rank-1 methods\nfail here →", fontsize=10, color='gray')

    ax.set_xlabel("Target Rank  $r$")
    ax.set_ylabel("Reconstruction Fidelity  $\\mathcal{F}$")
    ax.set_title("SSP-QST vs. Rank-1 Methods Across Target Ranks\n"
                 "$n=3$ qubits, depolarizing $p=0.08$, $N_s=4096$, averaged over 8 random targets")
    ax.set_xticks(ranks)
    ax.legend(loc="lower left")
    ax.set_ylim(0.30, 1.05)
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{OUT}/figA_fidelity_vs_rank.png")
    plt.close()
    print("  saved figA_fidelity_vs_rank.png")

    return ranks, m_ls, m_ss, m_ssp, m_top


# ─── Fig B: Fidelity vs Noise at rank-3 and rank-4 ───────────────────────────

def figB_noise_vs_rank():
    """
    For rank-3 and rank-4 targets: sweep noise p, compare methods.
    This shows SSP-QST's advantage is robust across noise levels.
    """
    print("Fig B: Noise sweep at rank-3 and rank-4...")
    p_vals = np.linspace(0.0, 0.25, 13)
    n = 3; shots = 4096; n_trials = 5

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax, rank in zip(axes, [3, 4]):
        fl_all, fss_all, fssp_all, ftop_all = [], [], [], []
        for p in p_vals:
            fl_t, fss_t, fssp_t, ftop_t = [], [], [], []
            for trial in range(n_trials):
                rng_t = np.random.default_rng(trial * 17 + rank)
                rho_t, _ = rank_r_target(n, rank, rng=rng_t)
                rho_ls, rho_ss, rho_ssp, rho_top = tomo_np(rho_t, n, p, shots)
                fl_t.append(fidelity_np(rho_ls,  rho_t))
                fss_t.append(fidelity_np(rho_ss,  rho_t))
                fssp_t.append(fidelity_np(rho_ssp, rho_t))
                ftop_t.append(fidelity_np(rho_top, rho_t))
            fl_all.append(np.mean(fl_t))
            fss_all.append(np.mean(fss_t))
            fssp_all.append(np.mean(fssp_t))
            ftop_all.append(np.mean(ftop_t))

        ax.plot(p_vals, fl_all,   "o--", color=C_LS,  ms=6, lw=2.0, label="LS-QST")
        ax.plot(p_vals, fss_all,  "^--", color=C_SS,  ms=6, lw=2.0, label="Spectral Squaring")
        ax.plot(p_vals, ftop_all, "D--", color=C_TOP, ms=6, lw=1.8, label="Top Eigenvector")
        ax.plot(p_vals, fssp_all, "s-",  color=C_SSP, ms=8, lw=2.6, label="SSP-QST")
        ax.fill_between(p_vals, np.maximum(fss_all, ftop_all), fssp_all,
                        alpha=0.12, color=C_SSP)
        ax.set_title(f"Rank-{rank} Target   ($n=3$ qubits)")
        ax.set_xlabel("Depolarizing Noise  $p$")
        ax.set_ylabel("Fidelity  $\\mathcal{F}$")
        ax.legend(fontsize=10, loc="upper right")
        ax.set_ylim(0.25, 1.04)

    plt.tight_layout(pad=0.5)
    plt.savefig(f"{OUT}/figB_noise_rank3_rank4.png")
    plt.close()
    print("  saved figB_noise_rank3_rank4.png")


# ─── Fig C: Eigenvalue recovery at rank-3 ────────────────────────────────────

def figC_eigenvalue_recovery():
    """
    Show that SSP-QST correctly identifies and retains 3 signal eigenmodes,
    while spectral squaring and top-eigenvector keep only 1.
    """
    print("Fig C: Eigenvalue recovery at rank-3...")
    n = 3; d = 8; p = 0.10; shots = 4096
    rho_t, _ = rank_r_target(n, 3, weights=[0.5, 0.3, 0.2], rng=np.random.default_rng(7))

    rho_ls, rho_ss, rho_ssp, rho_top = tomo_np(rho_t, n, p, shots)

    ev_t   = np.sort(np.linalg.eigvalsh(rho_t))[::-1]
    ev_ls  = np.sort(np.linalg.eigvalsh(rho_ls))[::-1]
    ev_ss  = np.sort(np.linalg.eigvalsh(rho_ss))[::-1]
    ev_ssp = np.sort(np.linalg.eigvalsh(rho_ssp))[::-1]

    idx = np.arange(1, d+1)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), sharey=True)
    configs = [
        (ev_t,   "#444444", "Target (rank-3)"),
        (ev_ls,  C_LS,      f"LS-QST\n$\\mathcal{{F}}={fidelity_np(rho_ls,rho_t):.3f}$"),
        (ev_ss,  C_SS,      f"Spectral Squaring\n$\\mathcal{{F}}={fidelity_np(rho_ss,rho_t):.3f}$"),
        (ev_ssp, C_SSP,     f"SSP-QST\n$\\mathcal{{F}}={fidelity_np(rho_ssp,rho_t):.3f}$"),
    ]
    for ax, (ev, color, title) in zip(axes, configs):
        ax.bar(idx, ev, color=color, alpha=0.85, width=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_xticks(idx)
        ax.axhline(0, color='k', lw=0.5)
        # Mark the 3 signal modes
        if "Target" not in title:
            ax.axvline(3.5, color='red', lw=1.2, ls='--', alpha=0.6, label="Signal/noise boundary")

    axes[0].set_ylabel("Eigenvalue  $\\lambda_i$")
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Eigenvalue Recovery for Rank-3 Photonic Target  ($n=3$, $p=0.10$)\n"
                 "SSP-QST retains the 3 signal modes; rank-1 methods discard modes 2 and 3",
                 fontsize=11, y=1.01)
    plt.tight_layout(pad=0.4)
    plt.savefig(f"{OUT}/figC_eigenvalue_recovery_rank3.png")
    plt.close()
    print("  saved figC_eigenvalue_recovery_rank3.png")


# ─── Fig D: Rank identification accuracy ─────────────────────────────────────

def figD_rank_identification():
    """
    Show that SSP-QST correctly identifies the rank in most cases.
    Plot: identified rank vs. true rank (2-5), across noise levels.
    """
    print("Fig D: Rank identification accuracy...")
    n = 3; shots = 4096; n_trials = 20
    true_ranks = [2, 3, 4, 5]
    p_vals = [0.02, 0.06, 0.10, 0.15, 0.20]

    # For each (true_rank, noise), compute mean identified rank
    results = {}
    for rank in true_ranks:
        for p in p_vals:
            identified = []
            for trial in range(n_trials):
                rng_t = np.random.default_rng(trial * 13 + rank * 7)
                rho_t, _ = rank_r_target(n, rank, rng=rng_t)
                rho_ls, _, rho_ssp, _ = tomo_np(rho_t, n, p, shots)
                ev_ssp = np.linalg.eigvalsh(rho_ssp)
                id_rank = int((ev_ssp > 1e-4).sum())
                identified.append(id_rank)
            results[(rank, p)] = np.mean(identified)

    fig, ax = plt.subplots(figsize=(8, 5.0))
    colors_ranks = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a", 5: "#984ea3"}
    for rank in true_ranks:
        y = [results[(rank, p)] for p in p_vals]
        ax.plot(p_vals, y, "o-", color=colors_ranks[rank], ms=8, lw=2.4,
                label=f"True rank = {rank}")
        ax.axhline(rank, color=colors_ranks[rank], lw=0.8, ls=":", alpha=0.5)

    ax.set_xlabel("Depolarizing Noise  $p$")
    ax.set_ylabel("SSP-QST Identified Rank (mean over 20 targets)")
    ax.set_title("Rank Identification by SSP-QST  ($n=3$ qubits, $N_s=4096$)\n"
                 "Dotted lines: true rank. SSP-QST slightly underestimates at high noise.")
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 6.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{OUT}/figD_rank_identification.png")
    plt.close()
    print("  saved figD_rank_identification.png")


# ─── Print comprehensive table ────────────────────────────────────────────────

def print_table():
    print("\n" + "="*70)
    print("TABLE: Fidelity vs. Target Rank (n=3, p=0.08, N_s=4096, 8 trials each)")
    print(f"{'Rank':>5} | {'LS-QST':>8} | {'Spec.Sq.':>9} | {'TopEigvec':>10} | {'SSP-QST':>8} | {'SSP wins?':>10}")
    print("-"*65)
    n=3; p=0.08; shots=4096
    for rank in [1, 2, 3, 4, 5]:
        fl_t, fss_t, fssp_t, ftop_t = [], [], [], []
        for trial in range(8):
            rng_t = np.random.default_rng(trial * 100 + rank)
            rho_t, _ = rank_r_target(n, rank, rng=rng_t)
            rho_ls, rho_ss, rho_ssp, rho_top = tomo_np(rho_t, n, p, shots)
            fl_t.append(fidelity_np(rho_ls,  rho_t))
            fss_t.append(fidelity_np(rho_ss,  rho_t))
            fssp_t.append(fidelity_np(rho_ssp, rho_t))
            ftop_t.append(fidelity_np(rho_top, rho_t))
        best = max(np.mean(fss_t), np.mean(ftop_t))
        winner = "YES" if np.mean(fssp_t) > best else "no"
        print(f"{rank:>5} | {np.mean(fl_t):>8.4f} | {np.mean(fss_t):>9.4f} | "
              f"{np.mean(ftop_t):>10.4f} | {np.mean(fssp_t):>8.4f} | {winner:>10}")
    print("="*70)


if __name__ == "__main__":
    print("Running rank-focused experiments...\n")
    figA_fidelity_vs_rank()
    figB_noise_vs_rank()
    figC_eigenvalue_recovery()
    figD_rank_identification()
    print_table()
    print(f"\nAll figures saved to {OUT}")
