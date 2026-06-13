#!/usr/bin/env python
"""Lane A diagnostics: is A1's FoM3 3822 real information or over-confidence?

Three deliverables, all from EXISTING gate dumps/curves (no GPU, no retrain):
  1. TARP coverage plot (dim-3, HIGH tercile + all-tercile-pooled) for
     A1 (VMIM 10-d) vs A2 (K=8) vs pair2dq (K=10 raw) — visual over/under-coverage.
  2. Corner overlay at ONE matched val truth (same point across arms; pooled 3 seeds)
     — visual posterior width comparison with truth crosshairs.
  3. Over-confidence -> FoM3 decomposition: per-param width-inflation k_i from the SBC
     rank std, prod(k_i) = FoM3 inflation, FoM3_calibrated = FoM3_obs / prod(k_i).
     Answers 'would the over-confidence have to be very significant?' with a number.

SBC->k model: for a posterior too narrow by factor k_i in param i, the SBC rank
r = Phi((truth-mu)/(sigma/k_i)) with (truth-mu)~N(0,sigma^2) => r = Phi(N(0,k^2)).
We invert std(r) numerically for k. prod(k_i) inflates FoM3 (FoM3 ~ 1/prod sigma_i,
correlations preserved). Caveat printed: SBC std also absorbs bias/non-Gaussianity, so
prod(k_i) is an UPPER bound on width-over-confidence and FoM3_cal a LOWER bound on the
true calibrated power.
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

FC = ("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/"
      "flatsky_cross_2026_06")
OUT = f"{FC}/overnight_menu_2/lane_a_plots"
PNAMES = ["Om", "s8", "w0"]
ARMS = {  # label -> (gate_root, arm_key, color)
    "A1 VMIM 10-d (3822)": (f"{FC}/overnight_menu_2/gate_c", "A1_pair2d_vmim", "tab:red"),
    "A2 pair2d K=8 (2874)": (f"{FC}/overnight_menu_2/gate_c", "A2_pair2d_k8", "tab:green"),
    "pair2d K=10 raw (2794)": (f"{FC}/overnight_menu/gate_c", "pair2dq_nobnt", "tab:blue"),
}
SEEDS = (41, 42, 43)


# ---------------------------------------------------------------- TARP curves
def load_curve(gate_root, arm, terc, seed, dim=3):
    f = f"{gate_root}/tarp_drp/curves/tarp_curve_{arm}_{terc}_seed{seed}_dim{dim}.npz"
    if not Path(f).exists():
        return None
    z = np.load(f)
    return np.asarray(z["alpha"]), np.asarray(z["ecp_bootstrap"]).mean(0)


def pooled_curve(gate_root, arm, terc):
    """Mean ECP over seeds on a common alpha grid."""
    curves = [load_curve(gate_root, arm, terc, s) for s in SEEDS]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None, None, None
    a0 = curves[0][0]
    ecps = np.stack([np.interp(a0, c[0], c[1]) for c in curves])
    return a0, ecps.mean(0), ecps.std(0)


def tarp_plot():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, terc, title in zip(axes, ("HIGH", "ALL"),
                               ("HIGH-FoM3 tercile (tightest posteriors)",
                                "all terciles pooled")):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
        for label, (groot, arm, col) in ARMS.items():
            if terc == "ALL":
                # average the three tercile curves
                accum = []
                for t in ("HIGH", "MID", "LOW"):
                    a, m, _ = pooled_curve(groot, arm, t)
                    if a is not None:
                        accum.append((a, m))
                if not accum:
                    continue
                a0 = accum[0][0]
                m = np.mean([np.interp(a0, a, mm) for a, mm in accum], axis=0)
                s = None
            else:
                a0, m, s = pooled_curve(groot, arm, terc)
                if a0 is None:
                    continue
            ax.plot(a0, m, color=col, lw=1.8, label=label)
            if s is not None:
                ax.fill_between(a0, m - s, m + s, color=col, alpha=0.15)
        ax.set_xlabel("credibility level  α")
        ax.set_ylabel("expected coverage  ECP")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.97, 0.06, "below diagonal =\nover-confident", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("TARP-DRP coverage — joint one-point statistic, dim-3 science subspace",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/tarp_lane_a.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/tarp_lane_a.png/pdf", flush=True)


# ---------------------------------------------------------------- corner overlay
def gather_arm_points(gate_root, arm):
    """All val points for an arm: dict theta-key -> list of (samples (M,6)) over seeds."""
    pts = {}
    for seed in SEEDS:
        for terc in ("HIGH", "MID", "LOW"):
            for f in glob.glob(f"{gate_root}/tarp_drp/dumps/{arm}_{terc}/seed_{seed}/"
                               "n*_m*/posterior_samples.npz"):
                z = np.load(f)
                s, th = z["samples"], z["theta"]
                for i in range(th.shape[0]):
                    key = tuple(np.round(th[i].astype(np.float64), 8))
                    pts.setdefault(key, []).append(s[i])
    return pts


def corner_plot():
    from getdist import MCSamples
    from getdist import plots as gdplt
    # reference arm A1: pick a 'typical' truth (closest to prior-center) it tightly constrains
    a1_root, a1_arm, _ = ARMS["A1 VMIM 10-d (3822)"]
    a1_pts = gather_arm_points(a1_root, a1_arm)
    keys = np.array(list(a1_pts.keys()))
    center = np.array([0.26, 0.84, -1.0, 0.673, 0.96, 0.0493])  # ~prior center
    sci = [0, 1, 2]
    d = np.linalg.norm((keys[:, sci] - center[sci]) / np.array([0.115, 0.288, 0.462]),
                       axis=1)
    truth_key = tuple(keys[int(np.argmin(d))])
    truth = np.array(truth_key)
    print(f"corner truth (Om,s8,w0) = {truth[:3]}", flush=True)

    mcs, colors = [], []
    for label, (groot, arm, col) in ARMS.items():
        pts = gather_arm_points(groot, arm)
        # match by nearest theta key (bit-equal in principle; nearest for safety)
        kk = np.array(list(pts.keys()))
        j = int(np.argmin(np.linalg.norm(kk - truth, axis=1)))
        mk = tuple(kk[j])
        samp = np.concatenate(pts[mk], axis=0)[:, sci]   # pooled seeds, (Nsamp,3)
        mcs.append(MCSamples(samples=samp, names=PNAMES,
                             labels=[r"\Omega_m", r"\sigma_8", "w_0"], label=label))
        colors.append(col)
    g = gdplt.get_subplot_plotter(width_inch=7.5)
    g.settings.legend_fontsize = 9
    g.triangle_plot(mcs, PNAMES, filled=True, colors=colors,
                    legend_labels=list(ARMS.keys()))
    # truth crosshairs
    for a, t in [(0, truth[0]), (1, truth[1]), (2, truth[2])]:
        for ax in g.subplots[:, a]:
            if ax is not None:
                ax.axvline(t, color="k", ls=":", lw=0.8)
    for ext in ("png", "pdf"):
        g.export(f"{OUT}/corner_lane_a.{ext}")
    print(f"wrote {OUT}/corner_lane_a.png/pdf", flush=True)
    return truth


# ---------------------------------------------------------------- over-confidence calc
_K_GRID = np.linspace(0.6, 3.0, 481)
_Z = np.random.default_rng(0).normal(size=200000)
_STD_GRID = np.array([norm.cdf(k * _Z).std() for k in _K_GRID])  # built ONCE at import


def sbc_std_to_k(std_target):
    """Invert std(rank) for the width-inflation k (rank = Phi(N(0,k^2))).
    Uses the precomputed monotone std(k) grid."""
    return float(np.interp(std_target, _STD_GRID, _K_GRID))   # stds increasing in k


def sbc_from_dumps(gate_root, arm):
    rs = []
    for f in sorted(glob.glob(f"{gate_root}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/"
                              "posterior_samples.npz")):
        z = np.load(f)
        rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))
    return np.concatenate(rs, 0) if rs else None


def overconf_decomposition():
    L = ["# Lane A — is FoM3 3822 real information or over-confidence?", "",
         "Per-param width-inflation k_i inferred from the SBC rank std (rank = Φ(N(0,k²))); ",
         "FoM3 inflation = Π k_i (FoM3 ∝ 1/Π σ_i, correlations preserved); ",
         "FoM3_calibrated = FoM3_obs / Π k_i.", "",
         "CAVEAT: SBC std also absorbs bias / non-Gaussianity, so Π k_i is an UPPER bound on "
         "width over-confidence and FoM3_calibrated a LOWER bound on true calibrated power.",
         "", "| arm | FoM3_obs | SBC std (Om,s8,w0) | k (Om,s8,w0) | Π k | FoM3_calibrated |",
         "|---|---|---|---|---|---|"]
    fobs = {"A1_pair2d_vmim": 3822, "A2_pair2d_k8": 2874, "pair2dq_nobnt": 2794}
    res = {}
    for label, (groot, arm, _) in ARMS.items():
        r = sbc_from_dumps(groot, arm)
        if r is None:
            continue
        stds = r.std(0)[:3]
        ks = np.array([max(1.0, sbc_std_to_k(s)) for s in stds])  # clamp k>=1 (over-conf only)
        prodk = float(np.prod(ks))
        fcal = fobs[arm] / prodk
        res[arm] = (stds, ks, prodk, fcal)
        L.append(f"| {label} | {fobs[arm]} | "
                 + ",".join(f"{s:.3f}" for s in stds) + " | "
                 + ",".join(f"{k:.2f}" for k in ks) + f" | {prodk:.2f} | {fcal:.0f} |")
    L += ["", "Calibrated baselines for comparison: l1+product 2875 (gate-C clean, |dev|≤0.037), "
          "l1-auto 2405.", "",
          "Reading (derived): compare FoM3_calibrated to 2875. If above, real information "
          "survives the over-confidence correction; if ≈ or below, the boost is mostly "
          "over-confidence."]
    Path(OUT, "OVERCONFIDENCE_DECOMPOSITION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return res


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    overconf_decomposition()
    tarp_plot()
    corner_plot()
    print("LANE A PLOTS DONE", flush=True)


if __name__ == "__main__":
    main()
