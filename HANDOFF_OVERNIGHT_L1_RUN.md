# HANDOFF — Overnight L1 auto+cross run (Part 2, autonomous)

**Created**: 2026-05-28 ~22:25 by the Part-2 "run continuation" session.
**Context**: Andreas going to sleep; greenlit running the L1 side autonomously +
prepping (not running) the CNN side. flip A/B verdict still pending; he said
"proceed flip=False now."

## ✅ RESULTS — L1 auto+cross DONE + VERIFIED (2026-05-28 23:57)
18 posteriors (2 arms × 3 seeds × 3 perms), all (100000,6), finite, valid_fom3=True.
Fan-out confirmed as dedup cache hits ("Loaded cached L1 train/val datasets (metadata matches)").

σ values below are PERM-CONSISTENT (per-perm pooled over seeds, then perm-averaged — same
pooling as FoM3). An earlier version reported all-9-pooled σ (perm-mixing broadened them and
wrongly showed auto+cross as *worse*); fixed in `analyze_l1_autocross_definitive.py`.

| arm | FoM3 (pooled, perm-avg) | σ(Ωm) | σ(σ8) | σ(w0) |
|---|---|---|---|---|
| auto+cross full  | **34,949** | 0.0277 | 0.0430 | 0.1392 |
| auto+cross 70/30 | **25,808** | 0.0296 | 0.0444 | 0.1284 |
| auto-only full (baseline, flip=True)  | 10,452 | 0.0355 | 0.0478 | 0.1740 |
| auto-only 70/30 (baseline, flip=True) | 8,086  | 0.0370 | 0.0497 | 0.1654 |

**Cross-channel gain — the honest read (FoM3 overstates; trust 2D + σ):**
- Marginal σ tightening (cross vs auto): Ωm ~1.25–1.28×, σ8 ~1.11×, w0 ~1.25–1.29×.
- 2D FoM gain: (Ωm,σ8) ~1.9–2.1×, (Ωm,w0) ~1.39×, (σ8,w0) ~1.34–1.39×.
- FoM3: 3.34× (full) / 3.19× (70/30) — real but compounded/correlation-amplified.
- auto+cross split penalty (full→70/30): 26%.
- ⚠️ **Flip-inconsistent**: auto+cross=flip=False, baselines=flip=True (flip=False ~21% higher
  FoM3 at s41/p0) → flip-consistent gain is somewhat lower, but the ~1.25× Ωm σ-tightening is
  robust enough to survive. Clean fix = auto-only flip=False re-run (route decision below).
- Figures: `figures/definitive_l1/{autocross_full_vs_split70, gain_fulltrain_*, gain_split70_*}.{pdf,png}`.
- Numbers: `DEFINITIVE_L1_SUMMARY.md` (now has a 2D-FoM table + gain breakdown), `definitive_l1_fom3.csv`.

## ⚠️ DECISIONS NEEDED FROM ANDREAS (CNN side blocked on these)
1. **Auto-only flip=False re-run (for an apples-to-apples gain).** Blocked on a route choice:
   the existing auto-only baselines use the **TFDS route** (`cross_maps=false`, 4ch); the cross
   script's `--channel-mode` only offers `auto_cross`/`cross_only` (NO `auto_only`). Options:
   (a) add an `auto_only` channel-mode to the cross script → route-matched 4ch-from-harmonic-cache
   auto-only, flip=False (cleanest, but a code change to the cross script); (b) re-run via the
   original TFDS auto-only script with flip=False (if it supports the flag); (c) accept flip=True
   baselines + document. I did NOT pick one (code change / methodology = your call).
2. **CNN compressor regime — RESOLVED 2026-05-29 (my earlier "asymmetry/ill-posed" claim was WRONG).**
   The fast harmonic-TFRecord path serves BOTH pipelines: auto+cross via `--channel-mode auto_cross`
   (10ch), auto-only via `--channel-mode auto_only` (slices to the 4 auto channels). Code-verified:
   `npe_cnn_nbody_tomo.py:4012` picks `build_harmonic_tfrecord_iterator` when `--harmonic-tfrecord-dir`
   is set (no `.npz` fallback); `:3689` auto_only→slice(0,4); `:3694` hard-errors if auto_only used off
   the harmonic route (can't slip to TFDS); `:3724` RMS norm computed post-slice (no shape mismatch).
   My error: I read the *existing* `auto_rnvp` compressor's route (TFDS) as a pipeline limitation — it
   was just an old compressor predating the fast path. Existing speeds: auto_rnvp TFDS 25.6 it/s (was
   already fast), autocross_rnvp harmonic-`.npz` 2.14 it/s (the slow one). Neither used `--harmonic-tfrecord-dir`.
   **Andreas-approved decision (2026-05-29): retrain BOTH from the harmonic TFRecord** with one
   consistent reconstructed config (defaults for the unrecorded compressor hyperparams), so all
   compressors share one route+shuffle regime. Launcher: `run_cnn_retrain_tfrecord.sh` (smoke gate →
   80k). New arms: `autocross_tfrec_rnvp` / `autoonly_tfrec_rnvp` (the latter is "cache-auto" — auto
   maps via SHT/iSHT roundtrip, slightly ≠ TFDS-direct auto; that's a deliberate sanity-check axis).
   Reconstructed config (logs+fiber+defaults): plain conv=(64,128,256) dense=256 pool=(16,8) cdim=10,
   compressor `train[:70%]` / NDE `train[70%:]`, 80k, save-every 1000, zero-mean, harmonic-normalize,
   best_val, compressor-lr 5e-4, compressor-batch 128, seed 41, RealNVP companion (script default).

## What changed vs the original handoff
The Part-1 session (commit `515004c`) **dropped the L1 TFRecord port** — an audit
found L1 summarization is wavelet-COMPUTE-bound, not read-bound, so TFRecord buys ≈0
for L1. The speedup is instead **cross-seed datavector dedup**: with `--no-l1-train-flip`
the train datavector is seed-independent (validated reproducible to 2.7e-11), so a
shared per-arm `--cache-dir` lets the 9 seed×perm runs share one ~50-min summarization.
So: **NO `--harmonic-tfrecord-dir` for L1.** The original §6 preconditions
(TFRecord flag + equivalence test) are superseded by Part-1's commit.

## What's running / will run (L1 side — fully autonomous)
- **Arm 1** `l1_autocross_fulltrain` (s41/p0, flip=False, `--nde-train-split train`) —
  launched on **GPU 0**, route confirmed healthy. ~138 patches/s.
- **Orchestrator** `scripts/sbi/orchestrate_l1_autocross_overnight.sh` (detached) drives:
  arm-2 warm-up `l1_autocross_split70` (`train[70%:]`) on GPU 1 when the F-leg frees it
  (fallback GPU 0) → wait for both datavector caches + warm-up posteriors → fan out the
  16 remaining runs across GPU 0+1 (cache hits) → run analysis.
- **18 posteriors** total (2 arms × 3 seeds {41,42,43} × 3 perms {0,1,2}).

## Files
- `scripts/sbi/run_l1_autocross_definitive.sh` — per-run launcher (modes: warmup_arm1/2,
  fanout_arm1/2, one). Common flags verbatim from the fiber §6 recipe + `--no-l1-train-flip`
  + shared `--cache-dir`. Uses `conda run --no-capture-output` + `PYTHONUNBUFFERED=1` so
  stdout streams (plain `conda run` block-buffers stdout — that's why the F-leg log sat
  empty for hours; the config markers only show with --no-capture-output).
- `scripts/sbi/orchestrate_l1_autocross_overnight.sh` — master sequencer (file-based polling,
  NO pkill).
- `scripts/sbi/analyze_l1_autocross_definitive.py` — FoM3 (perm-avg pooled = primary) +
  marginal σ + getdist overlays. VALIDATED: reproduces auto-only baselines 10,452 / 8,086.

## How to check status in the morning
```
cat scripts/sbi/results/exploratory/definitive_comparison/OVERNIGHT_STATUS.md       # live phase
ls  scripts/sbi/results/exploratory/definitive_comparison/.OVERNIGHT_L1_DONE         # exists => done
cat scripts/sbi/results/exploratory/definitive_comparison/DEFINITIVE_L1_SUMMARY.md   # numbers
ls  scripts/sbi/results/exploratory/definitive_comparison/figures/definitive_l1/     # corners
tail scripts/sbi/results/exploratory/definitive_comparison/logs/orchestrator.log
```

## flip A/B verdict — RESOLVED 2026-05-28 23:14 (s41/p0 full-train)
- flip=True (T-leg) = **FoM3 39,895**; flip=False (F-leg) = **FoM3 48,408**.
- flip=False is NOT worse (~+21% at this single seed) → the pause-trigger ("flip=True
  materially better") did NOT fire → **flip=False campaign proceeds as greenlit; dedup sound.**
- Caveats: (a) single-seed FoM3 is fragile — treat as flip≈neutral-to-mildly-favorable, not a
  hard 21%; (b) flip=False is *tighter* → possibly mildly optimistic → SBC/coverage check is the
  right follow-up (not done tonight).
- **Flip-consistency TODO (for Andreas / next session)**: the auto-only baselines (10,452/8,086)
  are flip=True. For a one-flip-setting definitive table, re-run auto-only **flip=False** (cheap,
  4ch) once the auto+cross campaign frees GPUs. Decision-rule note: this is now a *consistency*
  re-run, not a *correctness* one (flip=False isn't worse). I'll tee it up at L1-done.

## Expected anchors (sanity, don't force)
- L1 auto+cross full-train historically ~38k at perm 0 (v2 channel-aware noise). The T-leg
  flip=True hit 39,895. flip=False may differ.
- L1 auto+cross 70/30 is NEW — expect maybe ~24% below full (matching auto-only split
  penalty) but it's the open question. Measure it.

## CNN side (§7) — PREPPED, not run unsupervised
- Decision already approved by Andreas: run everything under the TFRecord regime → retrain
  the 2 `.npz`-trained RealNVP compressors on TFRecord (~1.3h each) so all compressors share
  one regime, then run RealNVP NDE arms (4,5,6,9) via Phase 0a.
- **Phase 0a** `scripts/sbi/train_jaxili_from_compressed.py` (NEW) — to draft per
  `~/.claude/plans/mighty-tumbling-sparrow.md` §0a; standalone jaxili MAF on compressed
  `cnn_{train,val,obs}.npz`.
- **Phase 0b** `--vmim-companion-backend maf` — edits SHARED CNN compressor-training code →
  needs Andreas sign-off + CNN test re-run. NOT done unsupervised. MAF arms (7,8,10) wait.

## Gotchas
- GPU policy (updated by Andreas 2026-05-28 evening): **max out GPU 0 AND GPU 1**; GPUs 2,3
  are other tenants — untouched. `XLA_PYTHON_CLIENT_PREALLOCATE=false` always.
- Never `git add -A`; stage by path. Don't delete the preserved debris dir.
- Claim success on measured FoM3/σ + the bit-exact baseline reproduction, never "ran without error".
