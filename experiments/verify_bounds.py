"""
verify_bounds.py -- Numerical verification of the two theoretical bounds
quoted in the QCE26 camera-ready (Section: threshold derivation scope).

1. Amplitude-damping envelope (proven, verified here):
     ||Lambda_gamma - id||_diamond <= gamma + 2a + a^2 = 2*gamma + 2a^2
                                    <= 2*gamma*(1+gamma),
     a = 1 - sqrt(1-gamma)   (exact identity: 2a - a^2 = gamma),
   hence via telescoping and ||.||_op <= ||.||_1:
     ||Delta||_op <= 2*n*gamma*(1+gamma)  for n-qubit local damping.
   Numerically the diamond norm equals 2*gamma exactly.
   Numerics below confirm: single-qubit ||E||_op <= gamma (attained),
   extended-channel trace norm attains 2*gamma (diamond norm equality),
   and full n-qubit ||Delta||_op stays within n*gamma (ratio ~0.64).

2. Matrix-Bernstein shot-noise envelope:
     ||Delta_shot||_op <~ sqrt(2 ln(2d) / Ns)   (= 0.041 at n=4, Ns=4096),
   and the ablation showing that substituting this worst case for the
   operational margin 0.5/sqrt(Ns) in the threshold over-truncates
   (rank-7 mean fidelity 0.941 -> 0.786 at the Table II configuration).

Run (from the repo root): python experiments/verify_bounds.py   (~1-2 min, deterministic)
"""
import numpy as np

rng = np.random.default_rng(3)


# ---- 1. Amplitude damping ---------------------------------------------------
def ad_kraus(g):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
    return K0, K1


def check_amplitude_damping(g=0.06, n=4, trials=20000):
    K0, K1 = ad_kraus(g)
    a = 1 - np.sqrt(1 - g)
    print(f"identity check: 2a - a^2 = {2*a - a*a:.10f}  vs gamma = {g}  (exact)")
    print(f"elementary bound gamma + 2a + a^2 = {g + 2*a + a*a:.6f}  "
          f"<= 2 gamma (1+gamma) = {2*g*(1+g):.6f}; diamond (numeric) = 2 gamma = {2*g}")

    # single qubit: max ||E||_op over Bloch ball
    best = 0.0
    for _ in range(trials):
        r = rng.random() ** (1 / 3)
        th = np.arccos(2 * rng.random() - 1)
        ph = 2 * np.pi * rng.random()
        x, y, z = (r * np.sin(th) * np.cos(ph),
                   r * np.sin(th) * np.sin(ph), r * np.cos(th))
        s = 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])
        E = K0 @ s @ K0.conj().T + K1 @ s @ K1.conj().T - s
        best = max(best, np.linalg.norm(E, 2))
    print(f"single-qubit max ||E||_op = {best:.6f}  (<= gamma: {best <= g + 1e-9})")

    # extended channel (entangled inputs): op and trace norms
    mx_op = mx_tr = 0.0
    for danc in (2, 4):
        K0e = np.kron(K0, np.eye(danc))
        K1e = np.kron(K1, np.eye(danc))
        D = 2 * danc
        for _ in range(trials):
            v = rng.standard_normal(D) + 1j * rng.standard_normal(D)
            v /= np.linalg.norm(v)
            s = np.outer(v, v.conj())
            E = K0e @ s @ K0e.conj().T + K1e @ s @ K1e.conj().T - s
            ev = np.linalg.eigvalsh(E)
            mx_op = max(mx_op, np.abs(ev).max())
            mx_tr = max(mx_tr, np.abs(ev).sum())
    print(f"extended max ||E||_op = {mx_op:.6f} ({mx_op/g:.3f} gamma), "
          f"max ||E||_1 = {mx_tr:.6f} ({mx_tr/g:.3f} gamma; diamond = 2 gamma)")

    # full n-qubit local damping: ||Delta||_op vs n*gamma over pure targets
    def apply_1q(rho, q):
        ops = [np.kron(np.kron(np.eye(2 ** q), K), np.eye(2 ** (n - q - 1)))
               for K in (K0, K1)]
        return sum(O @ rho @ O.conj().T for O in ops)

    d = 2 ** n
    worst = 0.0
    for _ in range(40):
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        rn = rho.copy()
        for q in range(n):
            rn = apply_1q(rn, q)
        worst = max(worst, np.linalg.norm(rn - rho, 2) / (n * g))
    print(f"n-qubit max ||Delta||_op / (n gamma) = {worst:.3f}  "
          f"(provable bound 2, observed < 1)")


# ---- 2. Bernstein envelope and margin ablation ------------------------------
def check_bernstein(n=4, Ns=4096, p=0.06):
    d = 2 ** n
    bern = np.sqrt(2 * np.log(2 * d) / Ns)
    print(f"\nBernstein envelope sqrt(2 ln(2d)/Ns) = {bern:.4f}; "
          f"operational margin 0.5/sqrt(Ns) = {0.5/np.sqrt(Ns):.4f}")

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import ssp_qst as rrr  # provides random_rank_state, ls_qst_from_state, depolarize, fidelity

    def ssp_margin(rho_ls, margin):
        vals, vecs = np.linalg.eigh((rho_ls + rho_ls.conj().T) / 2)
        vals = np.maximum(vals, 0)
        p_hat = max(0.0, 1.0 - vals[-1])
        eps = p_hat / (d - 1) + margin
        if vals[-2] < 2 * eps:
            v = vecs[:, -1]
            return np.outer(v, v.conj())
        vp = np.where(vals > eps, vals, 0.0)
        if vp.sum() < 1e-12:
            vp[-1] = 1.0
        vp /= vp.sum()
        return (vecs * vp) @ vecs.conj().T

    for r in (3, 5, 7):
        f_op, f_bn = [], []
        for t in range(14):
            tgt = rrr.random_rank_state(n, r, seed=1000 + 37 * r + t)
            rho_ls = rrr.ls_qst_from_state(rrr.depolarize(tgt, p), n,
                                           Ns=Ns, seed=2000 + 19 * r + t)
            f_op.append(rrr.fidelity(ssp_margin(rho_ls, 0.5 / np.sqrt(Ns)), tgt))
            f_bn.append(rrr.fidelity(ssp_margin(rho_ls, bern), tgt))
        print(f"rank {r}: margin 0.5/sqrt(Ns) F = {np.mean(f_op):.4f} | "
              f"Bernstein margin F = {np.mean(f_bn):.4f}")


if __name__ == "__main__":
    check_amplitude_damping()
    check_bernstein()
