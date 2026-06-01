# HANDOFF — CNN Phase A+B overnight (2026-05-30, fast tf.data route)

**For Andreas, morning of 2026-05-31.** Launched ~21:51 UTC after you said "max
out GPUs 0+1, run as much as possible overnight, do it the new [tf.data] way."

## RESULTS (as of ~02:35 UTC 05-31 — 9/10 posteriors done, autocross_cs43 pending)

`phaseB_tfdata_2026_05_30/SUMMARY_PHASEB.md`. Headline = compressor seed 41,
3 NDE seeds pooled, perm 0. **Leakage-flagged → trust the RELATIVE gain.**

- auto+cross FoM3 **29,120** vs auto-only **9,297** → **cross/auto = 3.13×**
- 2D FoM ratio: Ωm–σ8 **2.02×**, Ωm–w0 **1.99×**, σ8–w0 **1.77×**
- marginal σ tighten (auto/cross): w0 **1.45×**, Ωm **1.37×**, ns **1.42×**,
  h0 1.23×, σ8 1.16×, Ωb 1.03×
- compressor-seed variance (bonus, NDE s41): autocross cs42=20.5k (vs cs41
  pooled 29k); autoonly cs41/42/43 = 9.3k/11.9k/10.2k — sizeable, FoM3 is fragile
  ([[feedback_fom3_fragile_use_2d_areas]]) so weight the σ/2D metrics.

**Read:** the CNN cross-gain (~3.1× FoM3, ~2× 2D, ~1.2–1.45× σ) closely mirrors
the **L1** cross-gain on record (~3.2–3.3× FoM3, ~1.9–2.1× 2D, ~1.1–1.28× σ):
cross-map information helps L1 and CNN **comparably in relative terms.**
ABSOLUTE: CNN auto+cross 29.1k is leakage-inflated (~1.6× → de-inflated ~18k,
below clean L1 70/30 25.8k) — but absolute is NOT a clean comparison here.

**Note:** an early false alarm was just my watcher/aggregator globbing one dir
level too shallow (NDE nests `posteriors/<arm>/<arm-label>/`). Fixed; pipeline
was fine throughout (all NDE rc=0).

## What's running (two detached, self-driving orchestrators)

1. **Phase A — compressor retrain** (`scripts/sbi/run_cnn_phaseA_tfdata_overnight.sh`)
   - 2 RealNVP CNN compressors on the **fast tf.data cross route** (the local
     copy `/home/tersenov/tensorflow_datasets`, ~14–17 it/s, GPU-fed):
     - `autocross` (auto+cross, 10 ch, `--channel-mode auto_cross`) → **GPU 0**
     - `autoonly`  (auto-only, 4 ch sliced from the *same* cross dataset,
       `--channel-mode auto_only`) → **GPU 1**  (route-matched, so the
       auto-vs-cross delta isn't confounded by route)
   - Seeds 41,42,43 each; `plain` 64,128,256 / dense 256 / cdim 10; 80k steps;
     best-val; `--exit-after-compress` → compressed summaries land in
     `phaseA_tfdata_2026_05_30/compressed/<arm>_s<seed>/cnn_{train,val,obs}.npz`.
   - Independent per-arm loops (no cross-arm wait) so neither GPU idles. The
     **autoonly seed-41 run pays a one-time ~1 h `.npz` RMS scan** for the [0:4]
     slice (§5 of the perf-resolved handoff); seeds 42/43 hit the cache.
   - ETA: autocross ~1.3 h/seed; autoonly ~2.7 h (s41 w/ scan) then ~1.3 h.
     **Phase A all-done ≈ +5.7 h (~03:40 UTC).** Markers: `.done_<arm>_s<seed>`,
     `.PHASEA_TFDATA_DONE`. Live: `phaseA_tfdata_2026_05_30/orchestrator.log`.

2. **Phase B — jaxili MAF NDE** (`scripts/sbi/run_cnn_phaseB_nde_waiter.sh`)
   - Polls Phase A markers; as each compressor finishes, trains the jaxili MAF
     NDE (`train_jaxili_from_compressed.py`, **validated end-to-end tonight** —
     the only fix it needed was absolute paths for orbax).
   - compressor s41 → NDE seeds 41,42,43 (plan-faithful 3-NDE-seed headline);
     compressor s42,s43 → NDE seed 41 (compressor-seed-variance bonus). **Perm 0
     only** (cnn_obs.npz holds one perm; multi-perm needs `--obs-files`).
   - Output: `phaseB_tfdata_2026_05_30/posteriors/<arm>/<label>_s<ndeseed>_p0.{npy,fom.json,meta.json}`.
     Markers `.ndedone_*`, `.PHASEB_TFDATA_DONE`. Live: `phaseB_tfdata_2026_05_30/waiter.log`.

## ⚠️ READ FIRST: leakage flag (you chose "flag, don't fix")

`phaseA_tfdata_2026_05_30/README_LEAKAGE.md`. The cross TFDS was built with
`pool.imap_unordered` (`tf_dataset_nbody_tomo_cross.py:144`) → example order ≠
sorted-file order. Compressor trains on tf.data `train[:70%]` (random 70%); NDE
reads `.npz train[70%:]` (sorted last 30%) → ~70% realization overlap →
**absolute FoM inflated ~1.6×.** So:
- **auto-vs-cross RELATIVE CNN gain from this batch IS fair** (both arms leak identically).
- **L1-vs-CNN ABSOLUTE numbers are NOT trustworthy** from this batch.
- Clean rerun = drop `--cross-tfdata-dir` (compressor trains from `.npz` via
  `build_harmonic_batch_iterator`, same sorted-file slicing as NDE → genuinely
  disjoint). Cost: ~2.2 it/s at 4 loader threads; needs a `loader_threads`
  passthrough (not wired) + the 50-CPU budget to recover speed. **Decision left to you.**

## What I did NOT touch (left for your sign-off)

- **MAF VMIM companion (Phase 0b, plan arms 7,8,10).** Shared-code edit to
  `npe_cnn_nbody_tomo.py` — not done autonomously. Tonight is RealNVP-companion only.
- The npz route / any loader code. No commits.

## Verify in the morning
- `phaseA .../orchestrator.log` shows DONE for all 6 (`autocross|autoonly` × s41,42,43).
- `phaseB .../waiter.log` shows NDE DONE; `.fom.json` files exist; eyeball
  auto+cross vs auto-only FoM3 (relative gain is the trustworthy signal).
- GPU sanity: `nvidia-smi` — only my jobs on GPU 0/1; GPU 1 was free tonight
  (L1 campaign finished), GPU 2 had another tenant.
