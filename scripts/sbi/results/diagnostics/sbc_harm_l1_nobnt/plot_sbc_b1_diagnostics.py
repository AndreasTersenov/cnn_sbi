#!/usr/bin/env python3
"""SBC diagnostic plots for harmonic-L1 no-BNT (B1 calibration check).

Reads ranks from `n1000_m2000_seed20260507/sbc_ranks.npz` and produces three
plots:

  1. sbc_b1_rank_histograms.{pdf,png}
     6-panel rank histograms (one per parameter) with 95% confidence band
     for uniformity (binomial). Shape diagnoses:
        - flat            → well calibrated
        - U-shape         → posteriors too narrow (overconfident)
        - ∩-shape (dome)  → posteriors too wide  (underconfident)
        - upward slope    → posteriors biased low  (truth tends to be in tail)
        - downward slope  → posteriors biased high

  2. sbc_b1_rank_ecdfs.{pdf,png}
     6-panel empirical CDF of normalized ranks vs identity. The 95% band is
     a single-parameter Kolmogorov band at the given sample size.

  3. sbc_b1_summary_table.{pdf,png}
     One-page summary: mean-rank z-score, χ² per parameter, p-value, verdict.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "n1000_m2000_seed20260507"
RANKS_NPZ = RUN_DIR / "sbc_ranks.npz"
METRICS_JSON = RUN_DIR / "sbc_metrics.json"

PARAM_NAMES  = ["Omega_m", "sigma_8", "w0", "h0", "n_s", "Omega_b"]
PARAM_LABELS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$h_0$", r"$n_s$", r"$\Omega_b$"]

# ─── Load ─────────────────────────────────────────────────────────────────────
ranks = np.load(RANKS_NPZ, allow_pickle=True)["ranks"]   # shape (N, 6) ints
metrics = json.loads(METRICS_JSON.read_text())
N_RANKS = int(metrics["n_rank"])
POSTERIOR_M = int(metrics["posterior_samples"])
N_BINS = int(metrics["n_bins"])
assert ranks.shape == (N_RANKS, 6)

# Normalized rank ∈ [0,1] (canonical SBC variable)
u = ranks / POSTERIOR_M

# ─── Per-parameter stats ──────────────────────────────────────────────────────
def per_param_stats():
    out = []
    for i, name in enumerate(PARAM_NAMES):
        r = ranks[:, i]
        # Chi-square uniformity test (20 bins)
        hist, _ = np.histogram(r, bins=N_BINS, range=(0, POSTERIOR_M + 1))
        expected = N_RANKS / N_BINS
        chi2 = ((hist - expected) ** 2 / expected).sum()
        dof = N_BINS - 1
        p_chi2 = 1.0 - stats.chi2.cdf(chi2, dof)
        # KS test against uniform[0,1] on normalized ranks
        ks_stat, ks_p = stats.kstest(u[:, i], "uniform")
        # Mean-rank z-score
        mean_z = (r.mean() - POSTERIOR_M / 2) / np.sqrt(((POSTERIOR_M + 1) ** 2 - 1) / 12 / N_RANKS)
        out.append({
            "name": name,
            "chi2": float(chi2),
            "chi2_p": float(p_chi2),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "mean_rank_z": float(mean_z),
            "hist": hist,
            "expected": float(expected),
        })
    return out


def verdict(chi2_p, ks_p):
    """Strict-ish: both tests must agree to declare uniform."""
    m = min(chi2_p, ks_p)
    if m > 0.05: return "OK", "tab:green"
    if m > 0.01: return "marginal", "tab:olive"
    if m > 1e-3: return "non-uniform", "tab:orange"
    return "strongly non-uniform", "tab:red"


stats_per_p = per_param_stats()

# ─── PLOT 1: rank histograms ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, st, label in zip(axes.flat, stats_per_p, PARAM_LABELS):
    hist = st["hist"]
    bins = np.arange(N_BINS + 1) * (POSTERIOR_M + 1) / N_BINS
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    # 95% binomial confidence band for uniform
    p_bin = 1.0 / N_BINS
    mu = N_RANKS * p_bin
    sigma = np.sqrt(N_RANKS * p_bin * (1 - p_bin))
    lo, hi = mu - 1.96 * sigma, mu + 1.96 * sigma
    color = "tab:blue"
    v_label, v_color = verdict(st["chi2_p"], st["ks_p"])
    ax.bar(centers, hist, width=width, color=color, alpha=0.7, edgecolor="navy", linewidth=0.4)
    ax.axhspan(lo, hi, color="gray", alpha=0.25, lw=0, label="95% uniform band")
    ax.axhline(mu, ls="--", color="black", lw=0.8, label="uniform expectation")
    ax.set_title(
        f"{label}   χ²={st['chi2']:.1f} (p={st['chi2_p']:.3f})   "
        f"KS p={st['ks_p']:.3f}   [{v_label}]",
        color=v_color, fontsize=10,
    )
    ax.set_xlim(0, POSTERIOR_M)
    ax.set_xlabel("rank")
    ax.set_ylabel("count")

axes.flat[0].legend(loc="lower right", fontsize=8)
fig.suptitle(
    f"SBC rank histograms — harmonic-L1 no-BNT  (N={N_RANKS} cosmologies, "
    f"M={POSTERIOR_M} posterior samples, {N_BINS} bins)\n"
    "Flat ⇒ calibrated · U-shape ⇒ overconfident · ∩-shape ⇒ underconfident · slope ⇒ biased",
    y=1.00, fontsize=12,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"sbc_b1_rank_histograms.{ext}", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: sbc_b1_rank_histograms.{{pdf,png}}")

# ─── PLOT 2: ECDFs vs identity ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
# Kolmogorov 95% band: |F̂(x) - x| ≤ 1.36/√N
ks_band = 1.36 / np.sqrt(N_RANKS)
x_grid = np.linspace(0, 1, 200)
for ax, st, name, label in zip(axes.flat, stats_per_p, PARAM_NAMES, PARAM_LABELS):
    i = PARAM_NAMES.index(name)
    sorted_u = np.sort(u[:, i])
    ecdf_y = np.arange(1, N_RANKS + 1) / N_RANKS
    v_label, v_color = verdict(st["chi2_p"], st["ks_p"])
    ax.fill_between(x_grid, x_grid - ks_band, x_grid + ks_band,
                    color="gray", alpha=0.25, lw=0, label="95% KS band")
    ax.plot(x_grid, x_grid, "k--", lw=0.8, label="uniform")
    ax.plot(sorted_u, ecdf_y, color="tab:blue", lw=1.3, label="ECDF")
    ax.set_title(
        f"{label}   KS={st['ks_stat']:.3f} (p={st['ks_p']:.3f})   [{v_label}]",
        color=v_color, fontsize=10,
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("normalized rank u")
    ax.set_ylabel("F̂(u)")
    ax.set_aspect("equal", adjustable="box")

axes.flat[0].legend(loc="lower right", fontsize=8)
fig.suptitle(
    f"SBC empirical CDFs — harmonic-L1 no-BNT  (N={N_RANKS} cosmologies)\n"
    "Inside band ⇒ uniform (calibrated).  Curve above identity ⇒ rank-low bias; below ⇒ rank-high bias.",
    y=1.00, fontsize=12,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"sbc_b1_rank_ecdfs.{ext}", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: sbc_b1_rank_ecdfs.{{pdf,png}}")

# ─── PLOT 3: summary table ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.0))
ax.axis("off")
rows = []
headers = ["parameter", "χ² (dof=19)", "p_χ²", "KS stat", "p_KS",
           "mean-rank z", "verdict"]
for st, lab in zip(stats_per_p, PARAM_LABELS):
    v_label, v_color = verdict(st["chi2_p"], st["ks_p"])
    rows.append([
        lab,
        f"{st['chi2']:.2f}",
        f"{st['chi2_p']:.3g}",
        f"{st['ks_stat']:.3f}",
        f"{st['ks_p']:.3g}",
        f"{st['mean_rank_z']:+.2f}",
        v_label,
    ])

tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
               cellLoc="center", colColours=["#e8e8e8"] * len(headers))
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1, 1.6)
# Color the verdict cell per-parameter
for i, st in enumerate(stats_per_p):
    _, v_color = verdict(st["chi2_p"], st["ks_p"])
    tbl[i + 1, len(headers) - 1].set_facecolor(v_color)
    tbl[i + 1, len(headers) - 1].set_text_props(color="white", fontweight="bold")

ax.set_title(
    f"SBC summary — harmonic-L1 no-BNT  (N={N_RANKS}, M={POSTERIOR_M})\n"
    "OK: both p>0.05  ·  marginal: p>0.01  ·  non-uniform: p>1e-3  ·  strongly non-uniform: p≤1e-3",
    fontsize=12,
)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"sbc_b1_summary_table.{ext}", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: sbc_b1_summary_table.{{pdf,png}}")

# ─── Console summary ──────────────────────────────────────────────────────────
print()
print(f"{'param':>10}  {'χ²':>7}  {'p_χ²':>7}  {'KS':>6}  {'p_KS':>7}  {'mean-z':>7}  verdict")
print("-" * 70)
for st, lab_text in zip(stats_per_p, PARAM_NAMES):
    v, _ = verdict(st["chi2_p"], st["ks_p"])
    print(f"{lab_text:>10}  {st['chi2']:7.2f}  {st['chi2_p']:7.3g}  "
          f"{st['ks_stat']:6.3f}  {st['ks_p']:7.3g}  {st['mean_rank_z']:+7.2f}  {v}")
