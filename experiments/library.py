"""
Core library: Heavy-Tailed FTRL algorithms for episodic MDPs.

Implements:
  - HT-FTRL-OM  (Algorithm 1, known transitions)
  - HT-FTRL-UOB (Algorithm 2, unknown transitions)
  - Pareto heavy-tailed loss generation
  - Occupancy polytope FTRL solver
  - Claim 5: alpha-moment verification + skipping estimator analysis

Reference: arXiv 2602.01295 — "Best-of-Both-Worlds for Heavy-Tailed MDPs"
"""

import numpy as np
from itertools import product as iter_product


# ---------------------------------------------------------------------------
# Pareto heavy-tailed losses
# ---------------------------------------------------------------------------

def pareto_sample(shape, min_val, tail_index, rng):
    """Sample from Pareto(tail_index) with scale min_val."""
    u = rng.uniform(0, 1, size=shape)
    return min_val / (u ** (1.0 / tail_index))


def gen_losses(S, A, H, T, alpha, sigma, tail_index, rng, stochastic=False):
    """
    Generate T timesteps of losses for an (S, A, H) MDP.

    Parameters
    ----------
    S, A, H : int
        States, actions, horizon.
    T : int
        Number of episodes.
    alpha : float
        Moment order in (1, 2].
    sigma : float
        Moment bound: E[|X|^alpha] <= sigma^alpha.
    tail_index : float
        Pareto tail index (> alpha).
    rng : numpy.random.Generator
    stochastic : bool
        If True, losses are i.i.d. (same distribution per timestep).
        If False, losses are arbitrary adversarial.

    Returns
    -------
    losses : ndarray, shape (T, H, S, A)
    """
    min_val = sigma * ((tail_index - alpha) / tail_index) ** (1.0 / alpha)
    if stochastic:
        base = np.zeros((H, S, A))
        for h in range(H):
            for s in range(S):
                for a in range(A):
                    base[h, s, a] = pareto_sample((), min_val, tail_index, rng) + a * 0.5
        return np.tile(base, (T, 1, 1, 1))
    else:
        losses = np.zeros((T, H, S, A))
        for t in range(T):
            for h in range(H):
                for s in range(S):
                    for a in range(A):
                        losses[t, h, s, a] = pareto_sample((), min_val, tail_index, rng)
        return losses


# ---------------------------------------------------------------------------
# MDP utilities
# ---------------------------------------------------------------------------

def build_mdp(S, A, H):
    """
    Build a deterministic cyclic MDP: s_next = (s + a) mod S.

    Returns P[h][s][a] = {s_next: probability}.
    """
    P = []
    for h in range(H):
        Ph = []
        for s in range(S):
            Pa = []
            for a in range(A):
                Pa.append({(s + a) % S: 1.0})
            Ph.append(Pa)
        P.append(Ph)
    return P


def compute_value(P, loss_t, pi, S, A, H):
    """Compute V^pi for a single timestep's losses."""
    V = np.zeros((H + 1, S))
    for h in reversed(range(H)):
        for s in range(S):
            Q = np.zeros(A)
            for a in range(A):
                Q[a] = loss_t[h, s, a]
                for s_next, prob in P[h][s][a].items():
                    Q[a] += prob * V[h + 1, s_next]
            V[h, s] = np.dot(pi[h, s], Q)
    return V[0, 0]


def compute_occupancy(P, pi, S, A, H):
    """Compute state-action occupancy rho[h,s,a] under policy pi."""
    rho = np.zeros((H, S, A))
    sp = np.zeros(S)
    sp[0] = 1.0
    for h in range(H):
        for s in range(S):
            if sp[s] < 1e-15:
                continue
            for a in range(A):
                rho[h, s, a] = sp[s] * pi[h, s, a]
        nsp = np.zeros(S)
        for s in range(S):
            for a in range(A):
                for s_next, prob in P[h][s][a].items():
                    nsp[s_next] += rho[h, s, a] * prob
        sp = nsp
    return rho


def get_policy(rho, S, A, H):
    """Convert occupancy to a deterministic policy."""
    pi = np.zeros((H, S, A))
    for h in range(H):
        for s in range(S):
            t = rho[h, s].sum()
            pi[h, s] = rho[h, s] / t if t > 1e-12 else np.ones(A) / A
    return pi


def find_optimal(P, loss_t, S, A, H):
    """Brute-force optimal value (enumerate deterministic policies)."""
    best_V = float('inf')
    for a_star in range(A):
        pi = np.zeros((H, S, A))
        for h in range(H):
            for s in range(S):
                pi[h, s, a_star] = 1.0
        best_V = min(best_V, compute_value(P, loss_t, pi, S, A, H))
    return best_V


# ---------------------------------------------------------------------------
# FTRL solver (enumerate occupancy polytope for small MDPs)
# ---------------------------------------------------------------------------

def solve_ftrl_enum(cum_loss, eta, alpha, P, S, A, H):
    """
    Solve the Tsallis-regularized FTRL problem over the occupancy polytope.

    max_rho  -<rho, cum_loss> - (1/eta) * ||rho||_{1/alpha}^{1/alpha}
    """
    cum = cum_loss.reshape(H, S, A)
    best_val, best_rho = -1e30, None
    for seq in iter_product(range(A), repeat=H * S):
        pi = np.zeros((H, S, A))
        idx = 0
        for h in range(H):
            for s in range(S):
                pi[h, s, seq[idx]] = 1.0
                idx += 1
        rho = compute_occupancy(P, pi, S, A, H)
        val = -np.sum(rho * cum) - (1.0 / eta) * np.sum(np.maximum(rho, 1e-15) ** (1.0 / alpha))
        if val > best_val:
            best_val, best_rho = val, rho.copy()
    return best_rho


# ---------------------------------------------------------------------------
# HT-FTRL-OM (Algorithm 1): Known transitions
# ---------------------------------------------------------------------------

def run_htftrl_om(P, losses, S, A, H, T, alpha, sigma, C=2.0, beta=1.0):
    """
    Run Algorithm 1 (HT-FTRL-OM) for T episodes.

    Returns
    -------
    regrets : list of length T, per-episode regret
    """
    cum_loss = np.zeros((H, S, A))
    regrets = []
    for t in range(T):
        eta_t = beta / (sigma * (t + 1) ** (1.0 / alpha))
        x_t = solve_ftrl_enum(cum_loss, eta_t, alpha, P, S, A, H)
        pi_t = get_policy(x_t, S, A, H)
        V_l = compute_value(P, losses[t], pi_t, S, A, H)
        V_o = find_optimal(P, losses[t], S, A, H)
        regrets.append(V_l - V_o)
        for h in range(H):
            for s in range(S):
                for a in range(A):
                    xt = max(x_t[h, s, a], 1e-12)
                    tau = C * sigma * (t + 1) ** (1.0 / alpha) * xt ** (1.0 / alpha)
                    b_skip = C ** (1 - alpha) * sigma * (t + 1) ** (1.0 / alpha - 1) * xt ** (1.0 / alpha - 1)
                    ell = losses[t, h, s, a]
                    ell_skip = ell if abs(ell) <= tau else 0.0
                    cum_loss[h, s, a] += ell_skip / xt - b_skip
    return regrets


# ---------------------------------------------------------------------------
# HT-FTRL-UOB (Algorithm 2): Unknown transitions
# ---------------------------------------------------------------------------

def run_htftrl_uob(P, losses, S, A, H, T, alpha, sigma, C=2.0, beta=1.0, D=1.0):
    """
    Run Algorithm 2 (HT-FTRL-UOB) for T episodes.

    Returns
    -------
    regrets : list of length T, per-episode regret
    """
    cum_loss = np.zeros((H, S, A))
    m_sa = np.zeros((H, S, A))
    m_sas = np.zeros((H, S, A, S))
    m_old = np.zeros((H, S, A))
    t_start = 0
    regrets = []
    P_hat = []
    for h in range(H):
        Ph = []
        for s in range(S):
            Ph.append([{s_next: 1.0 / S for s_next in range(S)} for _ in range(A)])
        P_hat.append(Ph)
    for t in range(T):
        epoch_len = max(1, t - t_start + 1)
        eta_t = beta / (sigma * epoch_len ** (1.0 / alpha))
        x_t = solve_ftrl_enum(cum_loss, eta_t, alpha, P_hat, S, A, H)
        pi_t = get_policy(x_t, S, A, H)
        V_l = compute_value(P, losses[t], pi_t, S, A, H)
        V_o = find_optimal(P, losses[t], S, A, H)
        regrets.append(V_l - V_o)
        rho_true = compute_occupancy(P, pi_t, S, A, H)
        for h in range(H):
            for s in range(S):
                for a in range(A):
                    if rho_true[h, s, a] > 0.2:
                        m_sa[h, s, a] += 1
                        for s_next in P[h][s][a]:
                            m_sas[h, s, a, s_next] += 1
        if t > 0 and np.any(m_sa >= np.maximum(1, 2 * m_old)):
            t_start = t + 1
            m_old = m_sa.copy()
            for h in range(H):
                for s in range(S):
                    for a in range(A):
                        tot = m_sa[h, s, a]
                        if tot > 0:
                            P_hat[h][s][a] = {sn: m_sas[h, s, a, sn] / tot for sn in range(S)}
        rho_est = compute_occupancy(P_hat, pi_t, S, A, H)
        u_t = np.minimum(rho_est * 1.2 + 0.01, 1.0)
        for h in range(H):
            for s in range(S):
                for a in range(A):
                    xt = max(x_t[h, s, a], 1e-12)
                    tau = C * sigma * epoch_len ** (1.0 / alpha) * xt ** (1.0 / alpha)
                    b_skip = C ** (1 - alpha) * sigma * epoch_len ** (1.0 / alpha - 1) * xt ** (1.0 / alpha - 1)
                    u_sa = max(u_t[h, s, a], 1e-12)
                    ell = losses[t, h, s, a]
                    ell_skip = ell if abs(ell) <= tau else 0.0
                    n_sa = max(m_sa[h, s, a], 1)
                    log_term = np.log(T * S * A * H + 1)
                    B_i = min(1.0, np.sqrt(2 * log_term / n_sa) + 14 * log_term / (3 * n_sa))
                    cum_loss[h, s, a] += (ell_skip / u_sa) - b_skip - D * B_i
    return regrets


# ---------------------------------------------------------------------------
# Claim 5: Alpha-moment + Skipping estimator audit
# ---------------------------------------------------------------------------

def audit_claim5(alpha=1.5, tail_index=3.0, sigma=1.0, N=1_000_000, seed=42):
    """
    Verify Claim 5:
      (a) Pareto losses satisfy E[|X|^alpha] <= sigma^alpha
      (b) Truncative nonnegativity: E[X * 1(|X|<=M)] >= 0 for nonnegative X
      (c) Skipping estimator controls heavy-tailed outliers

    Returns dict with all audit results.
    """
    rng = np.random.default_rng(seed)
    min_val = sigma * ((tail_index - alpha) / tail_index) ** (1.0 / alpha)
    samples = pareto_sample((N,), min_val, tail_index, rng)

    # (a) Alpha-moment check
    moment_ratio = float(np.mean(samples ** alpha) / sigma ** alpha)
    moment_bounded = bool(moment_ratio <= 1.05)

    # (b) Truncative nonnegativity
    rng3 = np.random.default_rng(seed)
    samples3 = pareto_sample((100_000,), min_val, tail_index, rng3)
    trunc_results = {}
    for M in [1.0, 5.0, 10.0, 50.0, 100.0]:
        val = float(np.mean(samples3 * (np.abs(samples3) <= M)))
        trunc_results[f"M={M}"] = {"value": val, "nonneg": bool(val >= 0)}

    # (c) Skipping estimator
    C_skip = 2.0
    skip_results = {}
    for T in [100, 500, 1000, 5000]:
        rng2 = np.random.default_rng(seed)
        samples_T = pareto_sample((T,), min_val, tail_index, rng2)
        x_sa = 0.1
        tau = C_skip * sigma * T ** (1.0 / alpha) * x_sa ** (1.0 / alpha)
        b_skip = C_skip ** (1 - alpha) * sigma * T ** (1.0 / alpha - 1) * x_sa ** (1.0 / alpha - 1)
        ell_skip = np.where(np.abs(samples_T) <= tau, samples_T, 0.0)
        ell_est = ell_skip / x_sa - b_skip
        skip_frac = float(np.mean(np.abs(samples_T) > tau))
        skip_results[str(T)] = {
            "tau": float(tau),
            "skip_fraction": skip_frac,
            "E_estimator": float(np.mean(ell_est)),
            "E_estimator_sq": float(np.mean(ell_est ** 2)),
        }

    return {
        "alpha": alpha,
        "tail_index": tail_index,
        "sigma": sigma,
        "N_samples": N,
        "moment_ratio": moment_ratio,
        "moment_bounded": moment_bounded,
        "truncative_nonnegativity": trunc_results,
        "skipping_estimator": skip_results,
    }
