import os, itertools, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

# ------------------
# Quantum utilities
# ------------------
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
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

def spectral_squaring(rho):
    r2 = rho @ rho
    return project_density(r2 / np.trace(r2))

def top_eigvec(rho):
    vals, vecs = np.linalg.eigh(project_density(rho))
    return density(vecs[:, -1])

def fidelity(rho, sigma):
    sr = sqrtm(rho)
    middle = sr @ sigma @ sr
    val = np.trace(sqrtm(middle))
    out = float(np.real(val * np.conj(val)))
    return min(max(out, 0.0), 1.0)

def jz_operator(n):
    ops = []
    for q in range(n):
        mats = [I2] * n
        mats[q] = Z
        ops.append(kron_all(mats))
    return 0.5 * sum(ops)

def qfi(rho, H, eps=1e-12):
    rho = project_density(rho)
    vals, vecs = np.linalg.eigh(rho)
    H_eig = vecs.conj().T @ H @ vecs
    F = 0.0
    for i in range(len(vals)):
        for j in range(len(vals)):
            denom = vals[i] + vals[j]
            if denom > eps:
                F += 2.0 * ((vals[i] - vals[j]) ** 2 / denom) * abs(H_eig[i, j]) ** 2
    return float(np.real(F))

# ------------------
# Style helpers
# ------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

styles = {
    'LS-QST': dict(marker='o'),
    'Spec. Sq.': dict(marker='s'),
    'Top Eigvec.': dict(marker='^'),
    'SSP-QST': dict(marker='D'),
}

# ------------------
# Main experiments
# ------------------
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

def ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def parameterized_ghz_density(theta):
    U = kron_all([ry(th) for th in theta])
    psi = U @ ghz_state(len(theta))
    return density(psi)

def reconstruct_for_theta(theta, n, Ns, p, seed, method):
    rho = depolarize(parameterized_ghz_density(theta), p)
    rho_ls = ls_qst_from_state(rho, n, Ns=Ns, seed=seed)
    if method == 'LS-QST':
        return rho_ls
    return ssp_qst(rho_ls, Ns=Ns)

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
def savefig_all(fig, base):
    fig.savefig(os.path.join(FIGDIR, base + '.png'), dpi=400)
    fig.savefig(os.path.join(FIGDIR, base + '.pdf'))
    plt.close(fig)

# Compute data
print('Computing figure data...')
ranks, rank_means, rank_stds = compute_rank_fidelity()
p_vals, gains_noise = compute_gain_vs_noise()
Nss, shot_means, shot_stds = compute_shot_efficiency()
true_ranks, rid_pvals, rid_mat = compute_rank_identification()
qfi_ranks, qfi_means = compute_qfi_recovery()
loop_hist = compute_closed_loop()
ct_vals, phot_means = compute_photonic_structured()

# Fig A: fidelity vs rank
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
    ax.errorbar(ranks, rank_means[m], yerr=rank_stds[m], capsize=3, label=m, **styles[m])
ax.fill_between(ranks, rank_means['Spec. Sq.'], rank_means['SSP-QST'], alpha=0.15)
ax.set_xlabel('True probe rank $r$')
ax.set_ylabel('Reconstruction fidelity $\mathcal{F}$')
ax.set_ylim(0.2, 1.03)
ax.set_xticks(ranks)
ax.set_title('Fidelity vs. Probe Rank ($n=4$, $p=0.06$, $N_s=4096$)')
ax.legend(ncol=2, frameon=False, loc='lower left')
ax.annotate('SSP-QST remains best\nfor all mixed-state ranks', xy=(5, rank_means['SSP-QST'][4]), xytext=(3.4, 0.78),
            arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8)
# data-driven difference arrows at r = 7 (values computed, never hardcoded)
_r7 = len(ranks) - 1
_g_top = rank_means['SSP-QST'][_r7] - rank_means['Top Eigvec.'][_r7]
_g_sq  = rank_means['SSP-QST'][_r7] - rank_means['Spec. Sq.'][_r7]
ax.annotate('', xy=(7, rank_means['Top Eigvec.'][_r7]), xytext=(7, rank_means['SSP-QST'][_r7]),
            arrowprops=dict(arrowstyle='<->', color='#9467bd', lw=1.4))
ax.text(7.06, (rank_means['SSP-QST'][_r7] + rank_means['Top Eigvec.'][_r7]) / 2,
        f'+{_g_top:.3f}', color='#9467bd', fontsize=9, fontweight='bold', va='center')
ax.text(7.06, rank_means['SSP-QST'][_r7] + 0.005, f'+{_g_sq:.3f}',
        color='#d62728', fontsize=9, fontweight='bold', va='bottom')
ax.set_xlim(0.6, 7.9)
savefig_all(fig, 'figA_fidelity_vs_rank')

# Fig B: gain panels
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
# left: labeled gain bars (values computed from the same data as Table II)
_w = 0.26
_cols = {'LS-QST': '#1f77b4', 'Spec. Sq.': '#2ca02c', 'Top Eigvec.': '#9467bd'}
for k, (baseline, label) in enumerate([('LS-QST','vs LS-QST'), ('Spec. Sq.','vs Spectral Squaring'), ('Top Eigvec.','vs Top Eigenvector')]):
    gain = np.array(rank_means['SSP-QST']) - np.array(rank_means[baseline])
    xs = np.array(ranks) + (k - 1) * _w
    axes[0].bar(xs, gain, width=_w, label=label, color=_cols[baseline])
    if baseline == 'Top Eigvec.':
        for x, g in zip(xs, gain):
            if g > 0.05:
                axes[0].text(x, g + 0.01, f'{g:.2f}', ha='center', fontsize=7,
                             color=_cols[baseline], fontweight='bold')
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_xlabel('True probe rank $r$')
axes[0].set_ylabel(r'Fidelity gain $\Delta \mathcal{F}$ (SSP-QST $-$ baseline)')
axes[0].set_title('SSP-QST Gains Across Ranks')
axes[0].set_xticks(ranks)
axes[0].legend(frameon=False, fontsize=7, loc='upper left')
# right: gain over strongest baseline vs p
for r in [3,5]:
    axes[1].plot(p_vals, gains_noise[r], marker='o', label=f'Rank {r}')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xlabel('Depolarizing noise $p$')
axes[1].set_ylabel('Gain over strongest baseline')
axes[1].set_title('Robust Positive Gain Across Noise')
axes[1].legend(frameon=False)
savefig_all(fig, 'figB_gain_panels')

# Fig C: shot efficiency
fig, ax = plt.subplots(figsize=(4.2, 3.3))
for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
    ax.errorbar(Nss, shot_means[m], yerr=shot_stds[m], capsize=3, label=m, **styles[m])
ax.set_xscale('log', base=2)
ax.set_xticks(Nss)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('Shots per Pauli basis $N_s$')
ax.set_ylabel('Reconstruction fidelity $\mathcal{F}$')
ax.set_ylim(0.4, 1.02)
ax.set_title('Shot Efficiency (Rank 3, $p=0.08$)')
ax.legend(frameon=False, fontsize=7, loc='lower right')
ax.annotate('SSP-QST at 512 shots\noutperforms all baselines', xy=(512, shot_means['SSP-QST'][0]), xytext=(700, 0.72),
            arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=7)
savefig_all(fig, 'figC_shot_efficiency')

# Fig D: rank identification
fig, ax = plt.subplots(figsize=(4.2, 3.4))
markers = ['o', 's', '^', 'D', 'v']
for idx, p in enumerate(rid_pvals):
    ax.plot(true_ranks, rid_mat[idx], marker=markers[idx], label=f'$p={p:.02f}$')
ax.plot(true_ranks, true_ranks, '--', color='black', lw=1.0, label='Ideal')
ax.set_xlabel('True probe rank')
ax.set_ylabel('Mean identified rank')
ax.set_xticks(true_ranks)
ax.set_yticks(list(range(1,8)))
ax.set_ylim(1.0, 6.5)
ax.set_title('Rank Identification Accuracy')
ax.legend(frameon=False, fontsize=7, loc='upper left')
savefig_all(fig, 'figD_rank_identification')

# Fig E: additional validation 2x2
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
fig.subplots_adjust(top=0.89, bottom=0.10, left=0.09, right=0.985, hspace=0.30, wspace=0.18)

# (a) QFI recovery
ax = axes[0, 0]
for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
    ax.plot(qfi_ranks, qfi_means[m], linewidth=2.2 if m == 'SSP-QST' else 1.9,
            markersize=6 if m == 'SSP-QST' else 5, label=m, **styles[m])
ax.set_xlabel('True probe rank $r$')
ax.set_ylabel('QFI agreement score')
ax.set_title('(a) QFI recovery', pad=10)
ax.set_xticks(qfi_ranks)
ax.set_ylim(0.65, 1.005)
ax.grid(True, alpha=0.22)

# (b) structured photonic probes
ax = axes[0, 1]
for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
    ax.plot(ct_vals, phot_means[m], linewidth=2.2 if m == 'SSP-QST' else 1.9,
            markersize=6 if m == 'SSP-QST' else 5, label=m, **styles[m])
ax.set_xlabel('Crosstalk admixture')
ax.set_ylabel('Reconstruction fidelity')
ax.set_title('(b) Structured photonic probes', pad=10)
ax.set_ylim(0.845, 0.965)
ax.grid(True, alpha=0.22)

# (c) closed-loop drift correction
ax = axes[1, 0]
loop_styles = {
    'LS-QST': dict(marker='o'),
    'SSP-QST': dict(marker='o'),
}
for m in ['LS-QST', 'SSP-QST']:
    ax.plot(range(len(loop_hist[m])), loop_hist[m], linewidth=2.2 if m == 'SSP-QST' else 1.9,
            markersize=4.0, label=m, **loop_styles[m])
ax.set_xlabel('Feedback iteration')
ax.set_ylabel('Fidelity to target')
ax.set_title('(c) Closed-loop drift correction', pad=10)
ax.set_ylim(0.845, 1.005)
ax.grid(True, alpha=0.22)
# annotate line ends instead of using an in-panel legend
ax.text(len(loop_hist['SSP-QST']) - 1.2, loop_hist['SSP-QST'][-1] - 0.003, 'SSP-QST',
        ha='right', va='top', fontsize=8.5)
ax.text(len(loop_hist['LS-QST']) - 1.2, loop_hist['LS-QST'][-1] + 0.003, 'LS-QST',
        ha='right', va='bottom', fontsize=8.5)

# (d) representative scenarios
ax = axes[1, 1]
scenarios = [
    ('Rank-3\nfidelity', {
        'LS-QST': rank_means['LS-QST'][2],
        'Spec. Sq.': rank_means['Spec. Sq.'][2],
        'Top Eigvec.': rank_means['Top Eigvec.'][2],
        'SSP-QST': rank_means['SSP-QST'][2],
    }),
    ('QFI\nagreement', {
        'LS-QST': qfi_means['LS-QST'][1],
        'Spec. Sq.': qfi_means['Spec. Sq.'][1],
        'Top Eigvec.': qfi_means['Top Eigvec.'][1],
        'SSP-QST': qfi_means['SSP-QST'][1],
    }),
    ('512-shot\nfidelity', {
        'LS-QST': shot_means['LS-QST'][0],
        'Spec. Sq.': shot_means['Spec. Sq.'][0],
        'Top Eigvec.': shot_means['Top Eigvec.'][0],
        'SSP-QST': shot_means['SSP-QST'][0],
    }),
    ('Structured\nGHZ', {
        'LS-QST': phot_means['LS-QST'][3],
        'Spec. Sq.': phot_means['Spec. Sq.'][3],
        'Top Eigvec.': phot_means['Top Eigvec.'][3],
        'SSP-QST': phot_means['SSP-QST'][3],
    }),
]
labels = [s[0] for s in scenarios]
methods = ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']
x = np.arange(len(labels))
width = 0.18
for i, m in enumerate(methods):
    vals = [s[1][m] for s in scenarios]
    ax.bar(x + (i - 1.5) * width, vals, width=width, label=m)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='x', pad=7)
ax.margins(x=0.07)
ax.set_ylim(0.0, 1.05)
ax.set_ylabel('Score')
ax.set_title('(d) Representative scenarios', pad=10)
ax.grid(True, axis='y', alpha=0.22)

# shared legend placed in reserved top margin so it does not overlap the titles
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.985), columnspacing=1.3, handletextpad=0.5,
           borderaxespad=0.0)

savefig_all(fig, 'figE_additional_validation')

# Text file with numerical summary
summary_path = os.path.join(FIGDIR, 'selected_results_summary.txt')
with open(summary_path, 'w') as f:
    f.write('Representative results where SSP-QST is the best-performing method\n')
    f.write('===============================================================\n\n')
    f.write('Fidelity vs rank (n=4, p=0.06, Ns=4096)\n')
    for i, r in enumerate(ranks):
        f.write(f'r={r}: LS={rank_means["LS-QST"][i]:.4f}, Sq={rank_means["Spec. Sq."][i]:.4f}, Top={rank_means["Top Eigvec."][i]:.4f}, SSP={rank_means["SSP-QST"][i]:.4f}\n')
    f.write('\nQFI recovery at rank 3 (agreement score):\n')
    idx = qfi_ranks.index(3)
    for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
        f.write(f'{m}: {qfi_means[m][idx]:.4f}\n')
    f.write('\nShot efficiency at Ns=512 (rank 3, p=0.08):\n')
    idx = Nss.index(512)
    for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
        f.write(f'{m}: {shot_means[m][idx]:.4f}\n')
    f.write('\nStructured GHZ fidelity at crosstalk=0.06:\n')
    idx = ct_vals.index(0.06)
    for m in ['LS-QST', 'Spec. Sq.', 'Top Eigvec.', 'SSP-QST']:
        f.write(f'{m}: {phot_means[m][idx]:.4f}\n')
print('Wrote', summary_path)
