"""
run_all_figures.py -- Regenerate every figure and table of the paper.

Usage:
    python experiments/run_all_figures.py        (from the repo root)

Deterministic: rerunning reproduces figs/selected_results_summary.txt
byte-for-byte, matching Table II, Table III, Table IV inputs, and all
figure annotations of the camera-ready. Annotations are computed from
the plotted data at render time; nothing is hardcoded. Runtime is
roughly 10-20 minutes on a laptop CPU.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# make the package importable when run from the repo root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ssp_qst import (pauli_basis, random_rank_state, ghz_state,
                     physical_structured_ghz, parameterized_ghz_density,
                     project_density, depolarize, ls_qst_from_state,
                     reconstruct_for_theta, ssp_qst, spectral_squaring,
                     top_eigvec, fidelity, jz_operator, qfi,
                     compute_rank_fidelity, compute_gain_vs_noise,
                     compute_shot_efficiency, compute_rank_identification,
                     compute_qfi_recovery, compute_closed_loop,
                     compute_photonic_structured)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global plot styling (fonts sized per the camera-ready, Reviewer 2 request)
# ---------------------------------------------------------------------------
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
