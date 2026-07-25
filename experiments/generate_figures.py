#!/usr/bin/env python3
"""
generate_figures.py — Generate Plotly figures for all 5 claims.

Reads results from results/ and writes interactive HTML figures to figures/.

Usage:
    python generate_figures.py
"""

import json
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Install plotly: pip install plotly")
    exit(1)

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def fig_claim1(data):
    """Adversarial HT-FTRL-OM: Regret vs T^{1/alpha}."""
    Ts = [d["T"] for d in data]
    regrets = [d["regret"] for d in data]
    ratios = [d["regret_over_T1a"] for d in data]
    T1a = [T ** (1.0 / 1.5) for T in Ts]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Regret vs T", "Regret / T^{1/α} (should be flat)"])
    fig.add_trace(go.Scatter(x=Ts, y=regrets, mode="lines+markers", name="Regret",
                             line=dict(color="#2563eb", width=3), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=T1a, y=regrets, mode="lines+markers", name="Regret vs T^{1/α}",
                             line=dict(color="#dc2626", width=3, dash="dash"), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Ts, y=ratios, mode="lines+markers", name="Ratio",
                             line=dict(color="#16a34a", width=3), marker=dict(size=8)), row=1, col=2)
    fig.update_layout(title="Claim 1: Adversarial HT-FTRL-OM (Theorem 4.1)",
                      height=400, width=900, showlegend=True,
                      template="plotly_white")
    fig.update_xaxes(title_text="T", row=1, col=1)
    fig.update_xaxes(title_text="T", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Regret", row=1, col=1)
    fig.update_yaxes(title_text="Regret / T^{1/α}", row=1, col=2)
    return fig


def fig_claim2(data):
    """Stochastic HT-FTRL-OM: Regret vs log(T)."""
    Ts = [d["T"] for d in data]
    regrets = [d["regret"] for d in data]
    ratios = [d["regret_over_logT"] for d in data]
    logTs = [__import__("math").log(T) for T in Ts]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Regret vs T", "Regret / log(T) (should be flat)"])
    fig.add_trace(go.Scatter(x=Ts, y=regrets, mode="lines+markers", name="Regret",
                             line=dict(color="#7c3aed", width=3), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=logTs, y=regrets, mode="lines+markers", name="Regret vs log(T)",
                             line=dict(color="#ea580c", width=3, dash="dash"), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Ts, y=ratios, mode="lines+markers", name="Ratio",
                             line=dict(color="#0891b2", width=3), marker=dict(size=8)), row=1, col=2)
    fig.update_layout(title="Claim 2: Stochastic HT-FTRL-OM (Theorem 4.1, self-bounding)",
                      height=400, width=900, showlegend=True,
                      template="plotly_white")
    fig.update_xaxes(title_text="T", row=1, col=1)
    fig.update_xaxes(title_text="T", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Regret", row=1, col=1)
    fig.update_yaxes(title_text="Regret / log(T)", row=1, col=2)
    return fig


def fig_claim3(data):
    """Adversarial HT-FTRL-UOB: Regret vs T^{1/alpha} + sqrt(T)."""
    Ts = [d["T"] for d in data]
    regrets = [d["regret"] for d in data]
    ratios = [d["regret_over_bound"] for d in data]
    bounds = [T ** (1.0 / 1.5) + T ** 0.5 for T in Ts]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Regret vs T", "Regret / (T^{1/α}+√T) (should be flat)"])
    fig.add_trace(go.Scatter(x=Ts, y=regrets, mode="lines+markers", name="Regret",
                             line=dict(color="#be123c", width=3), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bounds, y=regrets, mode="lines+markers", name="Regret vs Bound",
                             line=dict(color="#4338ca", width=3, dash="dash"), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Ts, y=ratios, mode="lines+markers", name="Ratio",
                             line=dict(color="#059669", width=3), marker=dict(size=8)), row=1, col=2)
    fig.update_layout(title="Claim 3: Adversarial HT-FTRL-UOB (Theorem 5.2)",
                      height=400, width=900, showlegend=True,
                      template="plotly_white")
    fig.update_xaxes(title_text="T", row=1, col=1)
    fig.update_xaxes(title_text="T", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Regret", row=1, col=1)
    fig.update_yaxes(title_text="Regret / Bound", row=1, col=2)
    return fig


def fig_claim4(data):
    """Stochastic HT-FTRL-UOB: Regret vs (log T)^2."""
    Ts = [d["T"] for d in data]
    regrets = [d["regret"] for d in data]
    ratios = [d["regret_over_logT2"] for d in data]
    logT2 = [__import__("math").log(T) ** 2 for T in Ts]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Regret vs T", "Regret / (log T)^2 (should be flat)"])
    fig.add_trace(go.Scatter(x=Ts, y=regrets, mode="lines+markers", name="Regret",
                             line=dict(color="#9333ea", width=3), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=logT2, y=regrets, mode="lines+markers", name="Regret vs (log T)^2",
                             line=dict(color="#d97706", width=3, dash="dash"), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Ts, y=ratios, mode="lines+markers", name="Ratio",
                             line=dict(color="#0284c7", width=3), marker=dict(size=8)), row=1, col=2)
    fig.update_layout(title="Claim 4: Stochastic HT-FTRL-UOB (Theorem 5.2, self-bounding)",
                      height=400, width=900, showlegend=True,
                      template="plotly_white")
    fig.update_xaxes(title_text="T", row=1, col=1)
    fig.update_xaxes(title_text="T", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Regret", row=1, col=1)
    fig.update_yaxes(title_text="Regret / (log T)^2", row=1, col=2)
    return fig


def fig_claim5(data):
    """Claim 5: Alpha-moment verification + skipping estimator."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Truncative Nonnegativity", "Skipping Estimator Variance"])

    Ms = [float(k.split("=")[1]) for k in data["truncative_nonnegativity"]]
    vals = [data["truncative_nonnegativity"][k]["value"] for k in data["truncative_nonnegativity"]]
    fig.add_trace(go.Bar(x=[f"M={m}" for m in Ms], y=vals,
                         marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in vals],
                         name="E[X·1(|X|≤M)]"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    Ts_skip = [int(k) for k in data["skipping_estimator"]]
    variances = [data["skipping_estimator"][str(t)]["E_estimator_sq"] for t in Ts_skip]
    fig.add_trace(go.Scatter(x=Ts_skip, y=variances, mode="lines+markers",
                             name="E[estimator²]", line=dict(color="#ea580c", width=3),
                             marker=dict(size=8)), row=1, col=2)
    fig.update_layout(title="Claim 5: Alpha-Moment & Skipping Estimator (Assumption 5.1)",
                      height=400, width=900, showlegend=True,
                      template="plotly_white")
    fig.update_xaxes(title_text="M (truncation threshold)", row=1, col=1)
    fig.update_xaxes(title_text="T", row=1, col=2)
    fig.update_yaxes(title_text="E[X · 1(|X| ≤ M)]", row=1, col=1)
    fig.update_yaxes(title_text="E[estimator²]", row=1, col=2)
    return fig


def main():
    figs = {
        "fig1_claim1": (fig_claim1, "claim1_adv_om.json", "data"),
        "fig2_claim2": (fig_claim2, "claim2_stoch_om.json", "data"),
        "fig3_claim3": (fig_claim3, "claim3_adv_uob.json", "data"),
        "fig4_claim4": (fig_claim4, "claim4_stoch_uob.json", "data"),
        "fig5_claim5": (fig_claim5, "claim5_moment.json", None),
    }
    for name, (func, fname, key) in figs.items():
        with open(RESULTS_DIR / fname) as f:
            raw = json.load(f)
        data = raw[key] if key else raw
        fig = func(data)
        fig.write_html(FIGURES_DIR / f"{name}.html")
        print(f"  Wrote {FIGURES_DIR / f'{name}.html'}")
    print("Done.")


if __name__ == "__main__":
    main()
