#!/usr/bin/env python3
"""Per-seed (UN-pooled) CNN posteriors at the representative obs — and the BEST seed per arm.

The campaign pools 3 MAF/NDE seeds (the CNN has a single compressor, seed 41). Pooling applies a
"haircut" (the pooled posterior is wider than the best single seed). This reloads the already-trained
MAF checkpoints (CNN 10-d summary reloads bit-exact; L1's 2000-d would truncate) and samples each
seed SEPARATELY at the typical fiducial obs, reports per-seed FoM3, and picks the best seed per arm.
Tests whether the no-cross-gain result survives without the pool haircut.

Output: cnn_phase/best_seed/CNN_BEST_SEED.md (+ per_seed.json) + best-seed corner plots.
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = "/mnt/home/tersenov/software/cnn_sbi"; SBI = f"{REPO}/scripts/sbi"
FC = Path(f"{SBI}/results/exploratory/flatsky_cross_2026_06")
CNNP = FC / "cnn_phase"
OUT = CNNP / "best_seed"
ARMS = [("none", "auto-only"), ("conv", "+conv"), ("product", "+product"), ("both", "+both")]
SEEDS = [41, 42, 43]
TYP = (16, 23)   # typical patch (perm, patch)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-visible-devices", default="1")
    ap.add_argument("--m-samples", type=int, default=4000)
    a = ap.parse_args()
    sys.path.insert(0, REPO); sys.path.insert(0, SBI)
    from train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax, jax.numpy as jnp
    from jaxili.inference import NPE
    from npe_l1norm_cross_jaxili_nbody_tomo import (
        preprocess_summaries, filter_zero_variance_bins,
        _resolve_latest_jaxili_checkpoint_dir, _normalize_jaxili_hparams_embedding_arrays)

    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    samples_best = {}
    for op, lab in ARMS:
        cdir = CNNP / f"cnn_{op}_s41/cache"
        tr = np.load(cdir / "cnn_train.npz"); va = np.load(cdir / "cnn_val.npz")
        x_tr_raw = tr["x"].astype(np.float64); x_va_raw = va["x"].astype(np.float64)
        tr_p, _, _, mean, std = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
        mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
        fz = np.load(CNNP / "fiducial_summaries" / f"fiducial_summaries_{op}.npz")
        row = int(np.where((fz["perm"] == TYP[0]) & (fz["patch"] == TYP[1]))[0][0])
        _, _, fid_p, _, _ = preprocess_summaries(
            x_tr_raw, x_va_raw[:1], fz["S"].astype(np.float64)[row:row+1],
            summary_transform="none", clip_value=None, mean=mean, std=std)
        x_obs = fid_p[:, mask].astype(np.float32)[0]
        xdim = int(mask.sum())

        per_seed = {}
        for seed in SEEDS:
            ckroot = CNNP / "representative_corner" / f"flat_{op}" / "ckpts" / f"s{seed}"
            vdir = _resolve_latest_jaxili_checkpoint_dir(ckroot)
            _normalize_jaxili_hparams_embedding_arrays(vdir)
            exmp = (jnp.zeros((1, 6), jnp.float32), jnp.zeros((1, xdim), jnp.float32))
            inf = NPE.load_from_checkpoints(checkpoint=str(vdir), exmp_input=exmp)
            post = inf.build_posterior()
            k = jax.random.PRNGKey(seed * 100003 + row)
            ps = np.asarray(post.sample(x=jnp.asarray(x_obs), num_samples=a.m_samples, key=k))
            ps = ps[np.all(np.isfinite(ps), 1)]
            f3 = compute_fom3(ps)["fom3"]; sg = list(marginal_stats(ps)["sigma"].values())
            per_seed[seed] = dict(fom3=float(f3), sigma=[float(s) for s in sg[:3]], samples=ps)
            print(f"[{op}] seed {seed}: FoM3={f3:.0f} sig(Om,s8,w0)={sg[0]:.3f},{sg[1]:.3f},{sg[2]:.3f}",
                  flush=True)
        best = max(SEEDS, key=lambda s: per_seed[s]["fom3"])
        samples_best[op] = per_seed[best]["samples"]
        table[op] = dict(label=lab, per_seed={s: dict(fom3=per_seed[s]["fom3"],
                         sigma=per_seed[s]["sigma"]) for s in SEEDS},
                         best_seed=best, best_fom3=per_seed[best]["fom3"],
                         best_sigma=per_seed[best]["sigma"])
        print(f"[{op}] BEST seed = {best} (FoM3={per_seed[best]['fom3']:.0f})\n", flush=True)

    np.savez(OUT / "best_seed_samples_typical.npz", **{op: samples_best[op] for op, _ in ARMS})
    json.dump(table, open(OUT / "per_seed.json", "w"), indent=2)

    # --- markdown table ---
    pooled = {op: json.load(open(CNNP / "population_sweep" / f"flat_{op}" / "median_summary.json"))["fom3"]
              for op, _ in ARMS}
    l1pool = {op: json.load(open(FC / "population_sweep" / f"flat_{op}" / "median_summary.json"))["fom3"]
              for op, _ in ARMS}
    L = ["# CNN best single (MAF) seed — un-pooled — at the typical obs (perm16/patch23)\n",
         "The CNN has ONE compressor (seed 41); the 3 pooled seeds are MAF/NDE seeds. Pooling haircuts "
         "the FoM3, so the best single seed is the CNN at its most favorable. Reloaded the trained MAF "
         "checkpoints (10-d reloads bit-exact). L1 pooled shown for reference (L1's 2000-d datavector "
         "can't be reloaded per-seed without a retrain).\n",
         "| arm | seed41 | seed42 | seed43 | **best** | (CNN pooled-median) | (L1 pooled-median) |",
         "|---|---|---|---|---|---|---|"]
    for op, lab in ARMS:
        ps = table[op]["per_seed"]
        L.append(f"| {lab} | {ps[41]['fom3']:.0f} | {ps[42]['fom3']:.0f} | {ps[43]['fom3']:.0f} | "
                 f"**{table[op]['best_fom3']:.0f}** (s{table[op]['best_seed']}) | "
                 f"{pooled[op]:.0f} | {l1pool[op]:.0f} |")
    ca = table["none"]["best_fom3"]
    L += ["", "**Best-seed vs-auto ratios (does the no-cross-gain survive un-pooled?):**",
          " | ".join(f"{lab}: {table[op]['best_fom3']/ca:.2f}×" for op, lab in ARMS), ""]
    (OUT / "CNN_BEST_SEED.md").write_text("\n".join(L))
    print(f"wrote {OUT}/CNN_BEST_SEED.md + per_seed.json + best_seed_samples_typical.npz")


if __name__ == "__main__":
    main()
