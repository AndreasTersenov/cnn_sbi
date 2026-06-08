#!/usr/bin/env python3
"""Investigate the L1 per-patch FoM3 anomaly (patch_idx 0 atypically low; population
median ~2x higher). NDE-FREE structural diagnostics:

 1. Patch geometry: is patch_idx 0's sky center special (pole / edge / isolated)?
 2. Per-patch-index structure: grouping S by patch_idx, is patch 0 an outlier in raw
    datavector amplitude / OOD-ness, or is the high-FoM3 a per-index pattern?
 3. OOD test (the decisive one): preprocess the fiducial datavectors the way the NDE sees
    them (log1p-zscore-clip + mask, fit on TRAIN), measure each patch's distance from the
    training distribution, and CORRELATE per-patch FoM3 (from step2) with OOD-ness. Strong
    positive corr => the high per-patch FoM3 is an extrapolation artifact for atypical
    patches (and patch 0 is just typical), NOT a real L1 advantage.
"""
import csv, json
from pathlib import Path
import numpy as np

DC = Path("results/exploratory/definitive_comparison")
FF = DC / "fiducial_full200"
FIDCACHE = Path("results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial")


def preprocess_fit(x_tr, X, transform="log1p-zscore", clip=5.0, min_var=1e-5):
    """log1p -> zscore(train) -> clip -> mask(train var). Returns x_tr_p, X_p (masked)."""
    tr = np.log1p(x_tr); Xl = np.log1p(X)
    mu = tr.mean(0); sd = tr.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    tr = np.clip((tr - mu) / sd, -clip, clip); Xl = np.clip((Xl - mu) / sd, -clip, clip)
    mask = tr.var(0) > min_var
    return tr[:, mask], Xl[:, mask]


def main():
    z = np.load(FF / "summaries" / "l1_autocross_S.npz")
    S = z["S"].astype(np.float64); perm = z["perm"]; patch = z["patch"]
    x_tr = np.load(DC / "compressed" / "l1_autocross_split70_dv" / "l1_train.npz")["x"].astype(np.float64)
    print(f"S{S.shape} train{x_tr.shape}; perms {perm.min()}-{perm.max()}, patches {patch.min()}-{patch.max()}")

    # ---- 1. geometry ----
    obs0 = np.load(FIDCACHE / "nobnt" / "obs" / "cosmo_fiducial_perm0.npz")
    centers = np.asarray(obs0["patch_centers"])  # (48,2) (lon,lat) or (theta,phi)
    print("\n## 1. patch geometry")
    print(f"  patch_centers shape {centers.shape}; patch0 center = {centers[0]}")
    # nearest-neighbour separation of patch0 vs others (great-circle-ish on the 2 cols)
    def angsep(a, b):
        # treat cols as (lon,lat) deg; if radians small this still ranks correctly
        la1, la2 = np.radians(a[1]), np.radians(b[1]); dlon = np.radians(a[0]-b[0])
        return np.degrees(np.arccos(np.clip(np.sin(la1)*np.sin(la2)+np.cos(la1)*np.cos(la2)*np.cos(dlon), -1, 1)))
    nn = []
    for i in range(len(centers)):
        d = sorted(angsep(centers[i], centers[j]) for j in range(len(centers)) if j != i)
        nn.append(d[0])
    nn = np.array(nn)
    print(f"  nearest-neighbour sep: patch0 {nn[0]:.2f} deg vs median {np.median(nn):.2f} "
          f"(min {nn.min():.2f}, max {nn.max():.2f}) -> patch0 {'OUTLIER' if abs(nn[0]-np.median(nn))>2*np.std(nn) else 'typical'}")
    print(f"  patch0 lat {centers[0,1]:.2f} vs lat range [{centers[:,1].min():.2f},{centers[:,1].max():.2f}]")

    # ---- preprocess (NDE input space) ----
    x_tr_p, S_p = preprocess_fit(x_tr, S)
    tr_mean = x_tr_p.mean(0)
    # OOD proxies per patch (on S_p)
    meanabs = np.abs(S_p).mean(1)
    frac_clip = (np.abs(S_p) >= 4.99).mean(1)
    eucl = np.linalg.norm(S_p - tr_mean, axis=1) / np.sqrt(S_p.shape[1])
    rawnorm = np.linalg.norm(S, axis=1)

    # ---- 2. per-patch-index structure ----
    print("\n## 2. per-patch-index structure (mean over perms)")
    by = {pi: np.where(patch == pi)[0] for pi in range(int(patch.max())+1)}
    eucl_idx = np.array([eucl[by[pi]].mean() for pi in sorted(by)])
    print(f"  OOD(eucl) by patch_idx: patch0 {eucl_idx[0]:.3f} vs median {np.median(eucl_idx):.3f} "
          f"[{eucl_idx.min():.3f},{eucl_idx.max():.3f}]")
    order = np.argsort(eucl_idx)
    print(f"  lowest-OOD indices: {order[:5].tolist()} ; highest-OOD: {order[-5:].tolist()}")
    print(f"  is patch0 the lowest-OOD index? {'YES' if order[0]==0 else 'rank '+str(int(np.where(order==0)[0][0]))+'/48'}")

    # ---- 3. OOD vs FoM3 correlation (decisive) ----
    print("\n## 3. per-patch FoM3 vs OOD-ness (from step2 per_patch_fom.csv)")
    rows = list(csv.DictReader(open(FF / "posteriors" / "l1_autocross" / "per_patch_fom.csv")))
    gi = np.array([int(r["patch_global_idx"]) for r in rows])
    f3 = np.array([float(r["fom3"]) for r in rows]); pat = np.array([int(r["patch"]) for r in rows])
    o_e = eucl[gi]; o_c = frac_clip[gi]; o_m = meanabs[gi]; o_n = rawnorm[gi]
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    print(f"  corr(FoM3, eucl-OOD)   = {corr(f3, o_e):+.2f}")
    print(f"  corr(FoM3, frac-clip)  = {corr(f3, o_c):+.2f}")
    print(f"  corr(FoM3, mean|z|)    = {corr(f3, o_m):+.2f}")
    print(f"  corr(FoM3, raw L1 norm)= {corr(f3, o_n):+.2f}")
    # patch0 vs population on OOD
    m0 = pat == 0
    if m0.any():
        print(f"  patch0 (n={m0.sum()}): FoM3 {np.median(f3[m0]):.0f}, eucl-OOD {np.median(o_e[m0]):.3f} | "
              f"pop: FoM3 {np.median(f3[~m0]):.0f}, eucl-OOD {np.median(o_e[~m0]):.3f}")
    # high vs low FoM3 tercile OOD
    q1, q3 = np.percentile(f3, [33, 67])
    lo, hi = f3 <= q1, f3 >= q3
    print(f"  low-FoM3 tercile eucl-OOD {np.median(o_e[lo]):.3f} vs high-FoM3 tercile {np.median(o_e[hi]):.3f}")
    verdict = ("HIGH FoM3 ~ HIGH OOD => extrapolation artifact (population 53k suspect; patch0 typical)"
               if corr(f3, o_e) > 0.3 else
               "FoM3 NOT explained by OOD => high per-patch FoM3 likely structural/real (needs #2 TARP)")
    print(f"\n  VERDICT: {verdict}")
    out = {"corr_fom3_eucl": corr(f3, o_e), "corr_fom3_fracclip": corr(f3, o_c),
           "corr_fom3_rawnorm": corr(f3, o_n),
           "patch0_eucl_rank": int(np.where(order == 0)[0][0]),
           "patch0_nn_sep": float(nn[0]), "nn_sep_median": float(np.median(nn)),
           "verdict": verdict}
    (FF / "patch_anomaly_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(f"\n  -> {FF/'patch_anomaly_diagnosis.json'}")


if __name__ == "__main__":
    main()
