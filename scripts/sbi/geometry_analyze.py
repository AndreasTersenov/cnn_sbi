#!/usr/bin/env python3
"""Analyze the geometry-resample grids (Threads 1-3) for L1 vs CNN auto+cross.

Input: geometry_map/<arm>/per_patch_grid.csv (from geometry_resample.py), balanced
48 patch-indices x K perms, with fom3 / sig_* / bias_* / pull_* / lat / lon per patch.

Thread 1 (geometry map): per-index aggregate (median/mean over perms) of FoM3, sig(w0),
  |bias|; corr vs latitude; sky scatter; overlaid L1-vs-CNN vs-latitude panels.
Thread 2 (variance decomposition): balanced one-way decomposition of total per-patch
  spread into BETWEEN-index (geometry, fixed sky) vs WITHIN-index/BETWEEN-perm
  (realization noise), reported as eta^2 (frac explained by geometry), L1 vs CNN.
Thread 3 (bias structure): per-index mean bias (Om,s8,w0) + pull vs latitude; is L1's
  center-wander geometry-correlated (structured) or random, vs CNN?

No GPU. Lead with sigma/2D/pull, not FoM3 (FoM3 cubes ~20-25% diffs); FoM3 shown in log10.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAM_KEYS = ["Omega_m", "sigma_8", "w_0", "h_0", "n_s", "Omega_b"]
FOM3_PARAMS = ["Omega_m", "sigma_8", "w_0"]


def load_grid(csv_path: Path) -> dict:
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    out: dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])  # e.g. valid_fom3 strings
    out["patch"] = out["patch"].astype(int)
    out["perm"] = out["perm"].astype(int)
    return out


def per_index(g: dict, col: str, reducer=np.median) -> tuple[np.ndarray, np.ndarray]:
    """Return (patch_index_sorted, reduced_value_per_index) using only valid FoM3 rows if col=fom3."""
    idxs = np.unique(g["patch"])
    vals = []
    for pi in idxs:
        m = g["patch"] == pi
        x = g[col][m]
        if col == "fom3":
            v = g["valid_fom3"][m]
            keep = (v == "True") | (v == "1.0") | (v == 1.0) if v.dtype == object else (v != 0)
            x = x[keep] if keep.sum() > 0 else x
        vals.append(reducer(x))
    return idxs, np.array(vals)


def lat_of_index(g: dict) -> np.ndarray:
    idxs = np.unique(g["patch"])
    return np.array([g["lat"][g["patch"] == pi][0] for pi in idxs])


def lon_of_index(g: dict) -> np.ndarray:
    idxs = np.unique(g["patch"])
    return np.array([g["lon"][g["patch"] == pi][0] for pi in idxs])


def eta2_geometry(g: dict, y: np.ndarray) -> dict:
    """Balanced one-way variance decomposition of y over patch-index groups.
    eta2 = SS_between_index / SS_total = fraction of variance from geometry (fixed sky)."""
    idxs = np.unique(g["patch"])
    ybar = y.mean()
    ss_total = float(((y - ybar) ** 2).sum())
    ss_between = 0.0
    for pi in idxs:
        m = g["patch"] == pi
        ss_between += float(m.sum()) * (y[m].mean() - ybar) ** 2
    ss_within = ss_total - ss_between
    return {"eta2_geometry": ss_between / ss_total if ss_total > 0 else float("nan"),
            "frac_realization": ss_within / ss_total if ss_total > 0 else float("nan"),
            "ss_total": ss_total, "n_groups": int(len(idxs))}


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-dir", default="results/exploratory/definitive_comparison/"
                    "fiducial_full200/geometry_map")
    ap.add_argument("--arms", default="l1_autocross,cnn_autocross")
    args = ap.parse_args()
    GM = Path(args.geometry_dir)
    figdir = GM / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a.strip()]

    grids = {}
    for arm in arms:
        p = GM / arm / "per_patch_grid.csv"
        if not p.exists():
            print(f"  MISSING {p} — skipping {arm}"); continue
        grids[arm] = load_grid(p)
        n = len(grids[arm]["patch"]); ni = len(np.unique(grids[arm]["patch"]))
        print(f"[{arm}] {n} patches, {ni} indices, "
              f"{len(np.unique(grids[arm]['perm']))} perms")

    report = {"arms": {}, "thread1_corr": {}, "thread2_decomp": {}, "thread3_bias": {}}
    colors = {"l1_autocross": "C3", "cnn_autocross": "C0",
              "l1_autoonly": "C1", "cnn_autoonly": "C9"}

    # ---------- Thread 1: per-index aggregates + corr vs latitude ----------
    agg = {}
    for arm, g in grids.items():
        idxs = np.unique(g["patch"])
        lat = lat_of_index(g); lon = lon_of_index(g)
        _, fom3_med = per_index(g, "fom3", np.median)
        _, sigw0_med = per_index(g, "sig_w_0", np.median)
        _, sigom_med = per_index(g, "sig_Omega_m", np.median)
        absbias = {p: per_index(g, f"bias_{p}", lambda v: np.mean(np.abs(v)))[1] for p in FOM3_PARAMS}
        meanbias = {p: per_index(g, f"bias_{p}", np.mean)[1] for p in FOM3_PARAMS}
        meanpull = {p: per_index(g, f"pull_{p}", np.mean)[1] for p in FOM3_PARAMS}
        agg[arm] = dict(idxs=idxs, lat=lat, lon=lon, fom3_med=fom3_med,
                        sigw0_med=sigw0_med, sigom_med=sigom_med,
                        absbias=absbias, meanbias=meanbias, meanpull=meanpull)
        report["thread1_corr"][arm] = {
            "corr_log10fom3_vs_lat": corr(np.log10(fom3_med), lat),
            "corr_log10fom3_vs_abslat": corr(np.log10(fom3_med), np.abs(lat)),
            "corr_sigw0_vs_lat": corr(sigw0_med, lat),
            "corr_sigw0_vs_abslat": corr(sigw0_med, np.abs(lat)),
            "patch0_lat": float(lat[idxs == 0][0]) if 0 in idxs else None,
            "patch0_fom3_med": float(fom3_med[idxs == 0][0]) if 0 in idxs else None,
            "pop_fom3_med": float(np.median(g["fom3"][(g["valid_fom3"] != "False")
                                  if g["valid_fom3"].dtype == object else slice(None)])),
            "fom3_med_min": float(fom3_med.min()), "fom3_med_max": float(fom3_med.max()),
        }

    # Figure A: FoM3 vs latitude (per-index median)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for arm, d in agg.items():
        c = colors.get(arm, "k")
        ax[0].scatter(d["lat"], d["fom3_med"], c=c, label=arm, s=36, alpha=0.8)
        ax[1].scatter(d["lat"], d["sigw0_med"], c=c, label=arm, s=36, alpha=0.8)
        ax[2].scatter(d["lat"], d["sigom_med"], c=c, label=arm, s=36, alpha=0.8)
        if 0 in d["idxs"]:
            for k in range(3):
                xv = d["lat"][d["idxs"] == 0]
                yv = [d["fom3_med"], d["sigw0_med"], d["sigom_med"]][k][d["idxs"] == 0]
                ax[k].scatter(xv, yv, facecolors="none", edgecolors=c, s=160, linewidths=1.8)
    ax[0].set_yscale("log"); ax[0].set_ylabel("per-index median FoM3")
    ax[1].set_ylabel(r"per-index median $\sigma(w_0)$")
    ax[2].set_ylabel(r"per-index median $\sigma(\Omega_m)$")
    for a in ax:
        a.set_xlabel("patch center latitude [deg]"); a.legend(fontsize=8); a.grid(alpha=0.3)
    ax[0].set_title("circled = patch-0 (polar)")
    fig.tight_layout(); fig.savefig(figdir / "fom3_sigma_vs_latitude.png", dpi=130); plt.close(fig)

    # Figure B: sky scatter (lon,lat) colored by per-index median FoM3
    fig, axs = plt.subplots(1, len(agg), figsize=(6.5 * len(agg), 4.6), squeeze=False)
    for k, (arm, d) in enumerate(agg.items()):
        sc = axs[0, k].scatter(d["lon"], d["lat"], c=np.log10(d["fom3_med"]),
                               cmap="viridis", s=90)
        axs[0, k].set_title(f"{arm}: log10 median FoM3 over sky")
        axs[0, k].set_xlabel("lon [deg]"); axs[0, k].set_ylabel("lat [deg]")
        fig.colorbar(sc, ax=axs[0, k])
    fig.tight_layout(); fig.savefig(figdir / "sky_fom3_map.png", dpi=130); plt.close(fig)

    # ---------- Thread 2: variance decomposition ----------
    for arm, g in grids.items():
        valid = (g["valid_fom3"] != "False") if g["valid_fom3"].dtype == object else np.ones(len(g["fom3"]), bool)
        y_logfom3 = np.log10(g["fom3"][valid])
        d_logfom3 = eta2_geometry({"patch": g["patch"][valid]}, y_logfom3)
        d_sigw0 = eta2_geometry(g, g["sig_w_0"])
        d_sigom = eta2_geometry(g, g["sig_Omega_m"])
        report["thread2_decomp"][arm] = {
            "log10fom3": d_logfom3, "sig_w_0": d_sigw0, "sig_Omega_m": d_sigom}

    # Figure C: variance decomposition bar
    metrics = ["log10fom3", "sig_w_0", "sig_Omega_m"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / max(len(grids), 1)
    x = np.arange(len(metrics))
    for k, (arm, dd) in enumerate(report["thread2_decomp"].items()):
        vals = [dd[m]["eta2_geometry"] for m in metrics]
        ax.bar(x + k * width, vals, width, label=arm, color=colors.get(arm, "k"), alpha=0.85)
    ax.set_xticks(x + width * (len(grids) - 1) / 2); ax.set_xticklabels(metrics)
    ax.set_ylabel(r"$\eta^2$ = frac. variance from geometry (patch-index)")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Geometry (fixed sky) vs realization (perm) — higher = more geometry-driven")
    fig.tight_layout(); fig.savefig(figdir / "variance_decomposition.png", dpi=130); plt.close(fig)

    # ---------- Thread 3: bias structure ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, p in enumerate(FOM3_PARAMS):
        for arm, d in agg.items():
            c = colors.get(arm, "k")
            ax[k].scatter(d["lat"], d["meanbias"][p], c=c, label=arm, s=34, alpha=0.8)
            if 0 in d["idxs"]:
                ax[k].scatter(d["lat"][d["idxs"] == 0], d["meanbias"][p][d["idxs"] == 0],
                              facecolors="none", edgecolors=c, s=160, linewidths=1.8)
        ax[k].axhline(0, color="gray", lw=0.8)
        ax[k].set_xlabel("patch center latitude [deg]")
        ax[k].set_ylabel(f"per-index mean bias({p})"); ax[k].legend(fontsize=8); ax[k].grid(alpha=0.3)
    fig.suptitle("Thread 3 — per-index posterior-mean bias vs latitude (circled = patch-0)")
    fig.tight_layout(); fig.savefig(figdir / "bias_vs_latitude.png", dpi=130); plt.close(fig)

    for arm, d in agg.items():
        report["thread3_bias"][arm] = {}
        for p in FOM3_PARAMS:
            report["thread3_bias"][arm][p] = {
                "corr_meanbias_vs_lat": corr(d["meanbias"][p], d["lat"]),
                "corr_meanbias_vs_abslat": corr(d["meanbias"][p], np.abs(d["lat"])),
                "std_per_index_meanbias": float(np.std(d["meanbias"][p])),
                "corr_meanpull_vs_lat": corr(d["meanpull"][p], d["lat"]),
                "meanpull_overall": float(np.mean(d["meanpull"][p])),
            }

    # ---------- robustness: patch-0-excluded corrs + eta2 null ratio + systematic offset ----------
    report["robustness_excl_patch0"] = {}
    report["systematic_offset"] = {}
    for arm, g in grids.items():
        d = agg[arm]
        keep = d["idxs"] != 0
        report["robustness_excl_patch0"][arm] = {
            "corr_log10fom3_vs_lat_no_p0": corr(np.log10(d["fom3_med"][keep]), d["lat"][keep]),
            "corr_sigw0_vs_lat_no_p0": corr(d["sigw0_med"][keep], d["lat"][keep]),
            "corr_biasOm_vs_lat_no_p0": corr(d["meanbias"]["Omega_m"][keep], d["lat"][keep]),
            "corr_biasw0_vs_abslat_no_p0": corr(d["meanbias"]["w_0"][keep], np.abs(d["lat"][keep])),
        }
        # eta2 null baseline + excl-patch0 for log10fom3 (the only arm with a visible geometry bar)
        valid = (g["valid_fom3"] != "False") if g["valid_fom3"].dtype == object else np.ones(len(g["fom3"]), bool)
        y = np.log10(g["fom3"][valid]); pm = g["patch"][valid]
        N = int(valid.sum())
        n_groups = len(np.unique(pm))  # 48 (20deg) or 180 (10deg) -- read from the data
        null = (n_groups - 1) / (N - 1)
        e_all = report["thread2_decomp"][arm]["log10fom3"]["eta2_geometry"]
        k2 = pm != 0
        e_no = eta2_geometry({"patch": pm[k2]}, y[k2])["eta2_geometry"]
        report["thread2_decomp"][arm]["log10fom3"]["eta2_null"] = null
        report["thread2_decomp"][arm]["log10fom3"]["n_patch_groups"] = int(n_groups)
        report["thread2_decomp"][arm]["log10fom3"]["eta2_over_null_all"] = e_all / null
        n_groups_no0 = len(np.unique(pm[k2]))
        report["thread2_decomp"][arm]["log10fom3"]["eta2_over_null_no_patch0"] = \
            e_no / ((n_groups_no0 - 1) / (int(k2.sum()) - 1))
        # systematic offset (population mean bias + pull) for the 3 FoM3 params
        report["systematic_offset"][arm] = {
            p: {"mean_bias": float(np.mean(g[f"bias_{p}"])),
                "mean_pull": float(np.nanmean(g[f"pull_{p}"]))}
            for p in FOM3_PARAMS}

    (GM / "geometry_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nFigures -> {figdir}/  Report -> {GM/'geometry_report.json'}")


if __name__ == "__main__":
    main()
