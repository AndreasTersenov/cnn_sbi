---
name: Definitive L1 vs CNN comparison — 10 arms, all confounds eliminated
status: open
tags:
    - experiment
    - sbi
    - cnn
    - l1
    - definitive
created-at: 2026-05-27T21:17:28.520289295Z
outcome: 'OPEN. Original objective (definitive L1-vs-CNN comparison, 10 arms, confounds eliminated) is SUBSTANTIVELY SETTLED, with a corrected headline. Arc of corrections: (1) per-perm-POOL bug -> per-perm-AVERAGE; (2) perm-0 ''L1>=CNN auto+cross'' was a favorable-draw -> perm-matched CNN~L1; (3) THE BIG ONE (2026-06-02/03): moved from 3 fiducial obs to the FULL 200 realizations (9600 patches). The fixed campaign obs was patch-0 = the POLAR patch, atypically low-info for L1''s near-polar wavelets (CNN patch-insensitive). CORRECTED HEADLINE (typical obs patch, median over ~300 patches): L1 ~ CNN auto+cross with a SMALL L1 edge in w0/cross-maps (sigma(w0) x1.34, sigma(Om) x1.18, 2D x1.6; FoM3 x2.17 but FoM3 amplifies ~20-25% diffs); auto-only a TIE. Both calibrated (tight L1 verified via stratified varied-theta TARP). Folded into PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md (corrected headline at top; perm-averaged patch-0 tables demoted to historical). Dead ends removed: mean-datavector (OOD for L1), fixed-theta coverage (degenerate DRP / shrinkage-confounded Mahalanobis). Sub-fibers all closed except the NEW open [[understand-per-patch-structure-2026-06]] (next phase: diagnostics to understand WHY, using the large per-patch sample). See HANDOFF_PER_PATCH_DIAGNOSTICS_2026-06-03.md.'
---

## Objective

Produce the first fully controlled L1 vs CNN comparison: same NDE (jaxili
MAF), same split discipline, same dataset, same preprocessing — only the
compressor differs. 10 arms, 90 posteriors. Full plan in
`~/.claude/plans/mighty-tumbling-sparrow.md`.

## Primary metric

**3-seed pooled FoM3 on (Ωₘ, σ₈, w₀)**, per perm, then perm-averaged.
Secondary: 2D FoM per parameter pair, marginal σ per parameter (all 6).

## Done condition

All 90 posteriors (10 arms × 3 seeds × 3 perms) computed, analysis complete,
`SUMMARY.md` written with definitive comparison table. No plateau-stop
(fixed plan, not iterative). Wall-clock budget: ~3 days.

**TARP coverage gate (added 2026-05-31, Andreas — priority 1):** the comparison is
NOT done until every "final definitive" posterior arm has a **TARP expected-coverage
plot** (N=200 cosmologies × M=2000 samples convention; test ensemble = the held-out
`cnn_val.npz` (θ,x)). A tight posterior only counts if it is *calibrated*, not just
narrow — so coverage is reported alongside FoM3/σ/2D for L1 and CNN (RealNVP + MAF
companions). Tooling: `tarp_from_compressed.py` (reuses the `train_jaxili_from_compressed`
NDE; separate from it so it doesn't disturb running campaigns). This was previously
only an offhand "SBC is the proper follow-up" note in the flip=False entry; it is now
a formal done-condition item.

## Loop Status (live)

**[2026-06-03] ✅ ORIGINAL OBJECTIVE SETTLED + CLEAN STOP → next phase is [[understand-per-patch-structure-2026-06]].**
Read `HANDOFF_PER_PATCH_DIAGNOSTICS_2026-06-03.md` FIRST. The big step: moved from 3 fiducial obs to the
**FULL 200 realizations (9600 patches)** → more correct results AND stricter diagnostics. The campaign's
fixed obs was **patch-0 = the POLAR patch** (lat 88.5°), atypically low-info for L1's near-polar wavelets
(CNN is patch-insensitive) — that biased the original "CNN ≳ L1 auto+cross". **Corrected headline (typical
obs patch):** L1 ≈ CNN auto+cross with a small L1 edge in w0/cross-maps (σ(w0) ×1.34, σ(Ωm) ×1.18, 2D ×1.6;
FoM3 ×2.17 but FoM3 amplifies); **auto-only a tie**; tight L1 calibrated (stratified varied-θ TARP). Folded
into `PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md` (corrected headline at top; patch-0 perm-averaged tables
demoted to historical). Dead ends deleted: mean-datavector (OOD for L1), fixed-θ coverage (degenerate DRP /
shrinkage-confounded Mahalanobis). Clean artifact set in `fiducial_full200/` (see handoff). Nothing running;
nothing committed. **Next session:** large-sample diagnostics to understand WHY (geometry map, spread
decomposition, bias structure, w0, SBC) — see the fiber + handoff.

**[2026-06-02] 🌙 FULL-200 FIDUCIAL STUDY RUNNING overnight (detached, GPU1). Fiber [[fiducial-full200-meandv]].**
Andreas wants the fiducial obs extended from 3 perms to the full 200 realizations (9600 patches): step 1 =
posterior at the MEAN datavector (de-noised single-survey contour, NOT 200×-tighter); step 2 = per-patch
FoM3/σ distribution (real which-sky scatter). 6 arms (L1/CNN × auto/auto+cross + CNN std + CNN MAF). Build
done (200 perms, 9.6 GB); both summary extractors validated (CNN G1 4.7e-5, L1 G1 0.0 + calib MATCH); each
arm behind a G3 gate (reproduce campaign perm-0 FoM3 within 20%, else abort+skip — no silent garbage). Live:
`fiducial_full200/STATUS.log`; results `fiducial_full200/FIDUCIAL_FULL200_SUMMARY.md`. NOTE for cold-read:
this is the only thing in flight; on completion verify G3 verdicts and close the fiber with numbers.

**[2026-06-01] 🔁 PERM-AVERAGING REFINEMENT DONE — headline shift. Read [[finding-perm-averaging-overturns-l1-lead]].**
Fixed `aggregate_all_arms.py` to per-perm-AVERAGE (the declared metric) instead of perm-POOL.
**The perm-0 "L1 ≥ CNN on auto+cross" result does NOT survive perm-averaging** — it was a favorable
perm-0 draw. Matched 3-perm comparison (both perm-averaged): auto+cross FoM3 L1 25808 (±27%) vs CNN
28093 (±12%) → **CNN nominally ahead**; L1 keeps only a modest, perm-fragile σ(w0) edge (0.128 vs
0.143). CAVEAT: L1=harmonic vs CNN=tf.data route confound still uncontrolled. TARP for the 2 genuinely-new
arms (std, native-auto) launched; multi-perm TARP is redundant (same NDE as core RealNVP arms → not
re-dumped). Decision (Andreas 2026-06-01): **FoM3 stays the declared primary**, reported with per-perm
spread; σ/2D secondary (no constitution metric-stanza change). Remaining: (d) commit session code w/ OK.

**[2026-05-31] ✅ CAMPAIGN SUBSTANTIVELY COMPLETE — read `HANDOFF_DEFINITIVE_COMPARISON_2026-05-31.md` (root) FIRST.**
**Sub-fibers:** [[finding-patch-center-confound-g8]] · [[maf-companion-not-bottleneck]] · [[bug-multiperm-no-train-flag]] · [[finding-perm-averaging-overturns-l1-lead]] (all closed); [[refine-phase-c-perm-matched]] (closed 2026-06-01).
10 arms computed (jaxili MAF NDE all), TARP coverage (3-seed) for the 6 core arms, Phase C table
written (`results/exploratory/definitive_comparison/PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md`).
Key results (σ-based; FoM3 fragile):
- **Patch-center confound (G8) is REAL and large** — native-TFDS auto-only (FoM3 14969, σw0 0.148)
  ≫ harmonic-cache-sliced auto-only (9125, 0.216). ⟹ the CNN cross-gain is route-sensitive: ~1.8×
  over a fair auto-only, not the 2.93× over the lossy harmonic auto-only. **Quote cross-gain with this caveat.**
- **MAF companion is WORSE than RealNVP** (≤ across all 5 seed pairings) ⟹ companion is NOT the
  bottleneck. Sub-investigation CLOSED.
- **Standardization ~neutral** (doesn't destroy info). ~~**L1 ≥ CNN-RealNVP**, driven by **w₀**
  (σ 0.125 vs 0.151)~~ — **SUPERSEDED 2026-06-01:** that was perm-0 only; on the perm-averaged
  matched comparison **CNN ≳ L1 on FoM3/2D**, L1 keeps only a modest σ(w0) edge. See
  [[finding-perm-averaging-overturns-l1-lead]]. **All arms reasonably calibrated** (TARP, mildly over-confident).
- **Leakage empirically negligible** (Andreas) → fast-route absolute FoM is fine; **clean rerun
  DEPRIORITIZED** (and `.npz` loader is GIL-bound, can't be sped up).
Remaining iterations (none mid-flight): ~~(a) per-perm-AVERAGE the multi-perm arms~~ DONE 2026-06-01;
~~(b) TARP the new arms~~ DONE (std + native-auto; multi-perm redundant); ~~(c) write the G8 confound
into the summary~~ DONE; (d) commit session code w/ Andreas OK — **still pending**.
Deeper threads held for Andreas: 120k steps, the w₀ question, SBC.

**[2026-05-30] ⛔ CNN DATA-PATH RESOLVED — read `HANDOFF_PERF_REGRESSION_RESOLVED_2026-05-30.md`
BEFORE any CNN retrain.** The CNN ~1 it/s / ~0% GPU wall was **storage** (the cross dataset on
`/nas` is a FUSE `mergerfs` mount, ~100 MB/s for the random-read pattern), NOT threads / DLPack /
mem-fraction. **Fix: read the LOCAL copy** — `--cross-tfdata-dir /home/tersenov/tensorflow_datasets`
(NOT `/nas`) → ~15–19 it/s, 80k compressor ≈ ~1.5 h. A loader **normalization bug was fixed**
(`ad75511`; be on HEAD ≥ `676d407` — any pre-fix `--cross-tfdata-dir` run is scientifically
invalid). The old `--harmonic-tfrecord-dir` / Grain paths are **deleted**. ⚠️ `train[:70%]` /
`train[70%:]` disjointness is NOT preserved on the new path (example-slice vs file-slice →
compressor↔NDE leakage → inflated FoM; the in-run "overlap=0" audit is misleading). Corrected
command + a smoke tripwire (must hit ~15–19 it/s, else still on `/nas` or pre-fix) are in that doc,
which supersedes `HANDOFF_CNN_TFRECORD_PERF_REGRESSION.md`.

**PART 2 (run continuation) — LAUNCHED 2026-05-28 ~22:25, autonomous overnight.**
Andreas greenlit "proceed flip=False now" (flip A/B verdict still pending). State:
- Arm-1 warm-up `l1_autocross_fulltrain` s41/p0 (flip=False, full-train) running on
  **GPU 0** — route confirmed (`cross_noise_model=channel_empirical_global`,
  channel_scale ~10⁴× on cross ch, 10-ch auto_cross, pca off). Summarizing ~138 patches/s.
- Master orchestrator `scripts/sbi/orchestrate_l1_autocross_overnight.sh` (PID-detached)
  self-drives the rest: launches arm-2 `l1_autocross_split70` s41/p0 on GPU 1 when the
  F-leg frees it (falls back to GPU 0), waits for both datavector caches + warm-up
  posteriors, fans out the 16 remaining (3 seeds × 3 perms − warm-up) across GPU 0+1
  (cache hits, ~fast NDE), then runs analysis.
- Scripts: launcher `run_l1_autocross_definitive.sh`, analysis
  `analyze_l1_autocross_definitive.py` (FoM3 perm-avg-pooled primary + marginal σ +
  getdist overlays; VALIDATED — reproduces baselines 10,452/8,086 exactly).
- **flip A/B verdict (RESOLVED 2026-05-28 23:14, s41/p0 full-train):** flip=False=**48,408**
  vs flip=True=**39,895**. flip=False is NOT worse (~+21% single-seed) → pause-trigger
  ("flip=True materially better") did NOT fire → **flip=False campaign proceeds + dedup is
  sound**. Caveats: (a) single-seed FoM3 is fragile (read as flip≈neutral-to-mildly-favorable,
  not a hard 21%); (b) flip=False is *tighter* → possibly mildly optimistic coverage → SBC
  check is the proper follow-up (out of scope tonight). **Flip-consistency TODO**: auto-only
  baselines are flip=True; for a one-flip-setting definitive table, re-run auto-only flip=False
  (cheap, 4ch) after the auto+cross campaign frees GPUs. Snapshot: `flip_ab_Fleg.fom.json`.
- **Status files**: `OVERNIGHT_STATUS.md` (live phase), `.OVERNIGHT_L1_DONE` (marker),
  `DEFINITIVE_L1_SUMMARY.md` + `definitive_l1_fom3.csv` + `figures/definitive_l1/` (results),
  `logs/orchestrator.log`. Killed flip=True s41/p0 debris preserved in
  `posteriors/l1_autocross_fulltrain/_killed_run_debris_flipTrue/`.
- **CNN side (§7)**: NOT run unsupervised. Plan: retrain 2 RealNVP compressors on TFRecord
  (regime fix Andreas approved) + draft Phase 0a `train_jaxili_from_compressed.py`, then
  RealNVP NDE arms. Phase 0b (MAF companion, SHARED-code edit) left for Andreas sign-off.
- **Full overnight handoff**: `HANDOFF_OVERNIGHT_L1_RUN.md`.
- **✅ L1 auto+cross DONE + VERIFIED (23:57, ~89 min total).** 18 posteriors, all (100000,6)
  finite valid_fom3=True; fan-out = dedup cache hits (verified). **auto+cross full pooled FoM3
  = 34,949** (σ 0.0277/0.0430/0.1392 perm-consistent), **70/30 = 25,808** (σ 0.0296/0.0444/0.1284).
  **Honest cross gain (FoM3 overstates — trust 2D+σ per feedback_fom3_fragile_use_2d_areas (memory)):**
  marginal σ tighten ×1.25-1.28 (Ωm) /1.11 (σ8) /1.25-1.29 (w0); 2D FoM ×1.9-2.1 (Ωm,σ8); FoM3 3.34×/3.19×.
  Split penalty 26%. ⚠️ flip-INCONSISTENT (auto+cross flip=False vs baselines flip=True; flip=False
  ~21% higher single-seed) but Ωm tightening robust. NOTE: fixed a marginal-σ pooling bug (all-perm
  pooled → per-perm pooled) in `analyze_l1_autocross_definitive.py`. Results: `DEFINITIVE_L1_SUMMARY.md`
  (now with 2D-FoM table), `definitive_l1_fom3.csv`, `figures/definitive_l1/`.
- **CNN NOT run autonomously — 2 decisions for Andreas (see `HANDOFF_OVERNIGHT_L1_RUN.md`):**
  (1) auto-only flip=False re-run needs a route choice (add `auto_only` channel-mode to cross
  script vs TFDS auto-only script vs accept flip=True); (2) compressor regime is asymmetric —
  `auto_rnvp`=TFDS-4ch, `autocross_rnvp`=harmonic-10ch, so "retrain both on TFRecord" is ill-posed
  (only autocross is a candidate). Exact original configs recovered from `logs/phase_a_*_rnvp.log`.
  Phase 0a `train_jaxili_from_compressed.py` DRAFTED + compiles (grounded on real npz format; un-run).

**Auto-only flip=False re-run — LAUNCHED 2026-05-29 06:46 (Andreas-authorized).** Added an
`auto_only` `--channel-mode` to `npe_l1norm_cross_jaxili_nbody_tomo.py` (purely additive elif:
slice 0:nbins, l1_auto_boundary=nbins → all auto-SNR; verified config: `channel_mode=auto_only`,
`raw_summary=800`=5×40×4, auto-SNR [-13,13]; auto_cross/cross_only untouched; also fixed a cosmetic
channel-count label). Route-matched flip=False auto-only arms `l1_autoonly_{fulltrain,split70}`
(harmonic route, 4 auto ch, flip=False, dedup) via `run_l1_autoonly_definitive.sh` +
`orchestrate_l1_autoonly_overnight.sh`. **✅ DONE (06:46→07:18, 18/18 valid).** Route-matched
auto-only pooled FoM3: full **10,191** (σ 0.0368/0.0480/0.1855), 70/30 **8,774** — within ~3% of the
old TFDS/flip=True baselines (10,452/8,086), so route+flip barely moved auto-only ⇒ comparison robust.
**CLEAN apples-to-apples cross gain** (auto+cross vs route-matched auto-only, both harmonic+flip=False):
full **3.43×** FoM3 / σ-tighten Ωm 1.33×,σ8 1.12×,w0 1.33× / 2D (Ωm,σ8) 2.09×; 70/30 **2.94×** FoM3 /
σ Ωm 1.30× / 2D (Ωm,σ8) 1.96×. ⇒ flip-consistency caveat RESOLVED — clean gain (3.43/2.94×) brackets
the earlier inconsistent (3.34/3.19×); cross gain is real, strongest in Ωm and the Ωm–σ8 plane.
Figures `figures/definitive_l1/gain_{fulltrain,split70}_routematched.{pdf,png}`. (Note: pre-existing
latent `--help` argparse `%`-format bug in the L1 script, unrelated to runs — surfaced, not fixed.)

**Phase tracker:**
- [~] Phase 0a: Code — `train_jaxili_from_compressed.py` (NEW) — in progress (separate session)
- [~] Phase 0b: Code — MAF companion for VMIM (`--vmim-companion-backend maf`) — pending; tied to TFRecord work
- [x] Phase A (RealNVP compressors): auto-only + auto+cross both DONE + diagnosed (2026-05-28)
- [ ] Phase A-rest: MAF-companion compressors — needs 0b
- [x] Phase A.5 (partial): auto_rnvp + autocross_rnvp compressed datasets cached
- [~] Phase B (partial): L1 arms running. **L1 auto-only DONE (arms 3 + new split70).** L1 auto+cross (arms 1+2) running on GPU0+1 via main launch script.
- [ ] Phase B-rest: CNN NDE arms (4–10) — need Phase 0a code
- [ ] Phase C: Analysis + comparison table + figures + SUMMARY.md

**Early results landed (2026-05-28):**
- **L1 auto-only full-train**: pooled FoM3 10,452; per-seed 12,028 ± 2,387 (s41/42/43); σ(Om,s8,w0)=0.0355/0.0478/0.1740.
- **L1 auto-only 70/30** (new arm, completes the {auto,auto+cross}×{full,70/30} L1 matrix): pooled FoM3 8,086; per-seed 8,997 ± 1,952.
- **Split penalty ~23–25%** (pooled −23%, MoS −25%), concentrated in Om–s8; w0 flat. Comparable to CNN's ~18–24% split sensitivity → L1 and CNN roughly equally split-sensitive (single-seed s41 had overstated it at 31%).
- Both RealNVP-companion compressors diagnosed: auto+cross has better val loss (−12.86 vs −12.48) + stronger Om coupling (+0.89 vs +0.76) + less feature redundancy (0.86 vs 0.96 corr) but more overfit drift (+0.57 vs +0.15 nats).
- TFRecord/tf.data conversion of harmonic cache in progress (separate session) to fix the ~2.4 it/s GIL-bound bottleneck on auto+cross compressor training (target ~15 it/s). Spec: `scripts/sbi/HARMONIC_TFRECORD_IMPLEMENTATION_SPEC.md`.

**L1 TFRecord/dedup decision (2026-05-28, evening) — for the runs session; supersedes the "TFRecord port" expectation for the L1 arms:**

- **Audit (measured, GPU 1): the L1 summarization is wavelet-COMPUTE-bound, NOT read-bound.** Per realization: `.npz` read ~83 ms (threaded prefetch, hidden behind compute) vs wavelet compute ~293 ms (10ch). TFRecord block-read measured *slower* (~474 ms); a warm-cache run confirmed no read speedup. **→ The L1 TFRecord port is DROPPED — it buys ~0 for L1** (unlike CNN, which re-streams maps every training step and got 7.4×). **Do NOT add `--harmonic-tfrecord-dir` to the L1 §6 commands.**
- **The real L1 lever is the cross-seed datavector dedup.** The L1 train/val datavector depends only on (split, channel-config) — not seed/perm. Compute once per arm, reuse across the 9 seed×perm runs → ~one 2–4 h summarization/arm + 9× ~2-min NDE, instead of 9× full summarization (~6–9×).
- **Landed in `npe_l1norm_cross_jaxili_nbody_tomo.py` to make this exact (validated, not committed yet):**
  1. `--l1-train-flip` / `--no-l1-train-flip` flag (default True). The flip is **NOT a no-op** — it changes the datavector ~10% (measured; starlet boundary effects). `--no-l1-train-flip` removes it → seed-independent. **Provisional decision: flip=False for the campaign**, pending the FoM A/B (below).
  2. Deterministic SNR-range calibration — the cross-channel reservoir now uses a fixed `torch.Generator` (it was unseeded → non-reproducible). channel-σ was already deterministic.
  3. Cache-metadata key now includes `nde_train_split` + `l1_train_flip` (fixes a latent split-collision bug; lets a shared `--cache-dir` dedup across seeds without serving the wrong split/flip).
- **Dedup-correctness gate PASSED** (`scripts/sbi/tests/test_l1_dedup_seed_independence.py`): with `--no-l1-train-flip` + the calibration fix, the datavector is reproducible to 2.7e-11 (float64 noise; ~14 orders below the flip's 10% effect) → cross-seed reuse is exact.
- **Flip A/B FoM verdict: PENDING.** flip=True leg done; flip=False leg still running (heavy evening GPU contention). Watcher will report FoM3/σ (T vs F). If flip matters at the FoM level, revert to flip=True and dedup is invalid (fall back to per-seed). Until then, proceed flip=False.

**Corrected §6 L1 auto+cross recipe (replaces the TFRecord version):**
- Keep `--full-sphere-cross-cache <.npz>` (obs + calibration + datavector source). **No `--harmonic-tfrecord-dir`.** Add `--no-l1-train-flip`.
- Use a **shared `--cache-dir` per arm** (NOT per-seed): e.g. `.../compressed/l1_autocross_fulltrain_dv` (arm 1, `--nde-train-split train`) and `.../compressed/l1_autocross_split70_dv` (arm 2, `--nde-train-split train[70%:]`). Per-run: `--seed`, `--harmonic-obs-perm`, `--save-dir`/`--posterior-out` per (seed,perm).
- **Orchestration:** run ONE (s41,p0) per arm FIRST to populate the shared datavector cache (the ~2–4 h summarization); confirm `cross_noise_model = channel_empirical_global` + channel_scale table + `pca_applied: False` in stdout; THEN fan the remaining 8 across GPU 0+1 — they hit the cache (metadata match, seed excluded) and only do NDE+sampling (~2 min each).
- Common flags unchanged from §6: `--zero-mean-maps --map-kind nbody --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 --pca-components 0 --l1-min-snr -13 --l1-max-snr 13 --cross-snr-percentile 1.0 --batch-size 256 --learning-rate 0.0001 --npe-samples 100000 --no-wandb --cross-noise-model channel_empirical_global --epochs 50000`.

**Next session unblocks**: Phase 0a (jaxili-from-compressed) → CNN NDE arms on the 2 finished compressors; then 0b + MAF compressors. For L1: the dedup path above is ready (pending the flip A/B FoM landing).

## Experiment arms

| Arm | Compressor | Input | Companion | NDE split | Std | Label | Phase A |
|-----|-----------|-------|-----------|-----------|-----|-------|---------|
| 1 | L1 wavelet | auto+cross 10ch | N/A | full-train | log1p-zscore | `l1_autocross_fulltrain` | — |
| 2 | L1 wavelet | auto+cross 10ch | N/A | 70/30 | log1p-zscore | `l1_autocross_split70` | — |
| 3 | L1 wavelet | auto-only 4ch | N/A | full-train | log1p-zscore | `l1_auto_fulltrain` | — |
| 4 | CNN-VMIM | auto-only 4ch | RealNVP | 70/30 | False | `cnn_auto_rnvp_nostd` | **overnight** |
| 5 | CNN-VMIM | auto+cross 10ch | RealNVP | 70/30 | False | `cnn_autocross_rnvp_nostd` | **overnight** |
| 6 | CNN-VMIM | auto+cross 10ch | RealNVP | 70/30 | True | `cnn_autocross_rnvp_std` | shares arm 5 compressor |
| 7 | CNN-VMIM | auto-only 4ch | MAF | 70/30 | False | `cnn_auto_maf_nostd` | needs 0b |
| 8 | CNN-VMIM | auto+cross 10ch | MAF | 70/30 | False | `cnn_autocross_maf_nostd` | needs 0b |
| 9 | CNN-VMIM | auto-only 4ch (cache) | RealNVP | 70/30 | False | `cnn_auto_harmcache_rnvp_nostd` | shares arm 4 compressor |
| 10 | CNN-VMIM | auto-only 4ch (cache) | MAF | 70/30 | False | `cnn_auto_harmcache_maf_nostd` | shares arm 7 compressor |

Arms 6, 9, 10 share compressors with other arms — only 4 compressor trainings
needed (auto-only RealNVP, auto+cross RealNVP, auto-only MAF, auto+cross MAF).

## Controlled variables

- NDE: jaxili MAF, hidden=[50,50], 5 layers, batch_size=256, lr=1e-4
- Compressor: plain CNN, 64,128,256, dense=256, cdim=10, 80k steps, best-val,
  save-every=1000 (increased from default 2000 for more best-val lottery tickets)
- zero_mean_maps: True
- Dataset: nonoverlap48 TFDS / harmonic cache
- pca_components: 0 (L1 arms)
- L1 cross noise model: channel_empirical_global (harmonic cache route only)

## Scientific questions this experiment answers

1. **Does CNN auto+cross match L1 auto+cross when NDE confound is removed?** Compare arms 5/8 vs arm 1.
2. **Does the VMIM companion quality matter?** Compare arms 4 vs 7 (auto-only) and 5 vs 8 (auto+cross).
3. **Does standardization help or hurt CNN auto+cross?** Compare arms 5 vs 6.
4. **Does L1 suffer from 70/30 split?** Compare arms 1 vs 2.
5. **Does the patch-center confound matter for auto-only?** Compare arms 4 vs 9 and 7 vs 10.
6. **Where exactly does the L1 advantage come from (if it survives)?** Compare marginal σ per parameter across all arms.
7. **Does CNN beat L1 on auto-only?** Compare arms 3 vs 4. The NDE-swap test hinted CNN auto-only has tighter w₀.

## Verification checklist (before declaring success)

1. All 90 posteriors have 100k samples, no NaN/Inf
2. All meta.json files record the full configuration (cross-check against manifest)
3. Marginal σ values are physically plausible (Ωm ~0.02-0.04, σ₈ ~0.03-0.05, w₀ ~0.10-0.20)
4. NDE training converged (check val loss curves — no NaN, no early divergence)
5. For 70/30 split arms: verify 0 example overlap in the split
6. For L1 arms: verify `pca_applied=False` and `cross_noise_model=channel_empirical_global`
7. Corner plot overlays are visually sensible (posteriors centered near truth, no pathological modes)

## Code changes required (Phase 0)

**READ the full implementation plan at `~/.claude/plans/mighty-tumbling-sparrow.md`
before writing any code.** It contains detailed specs for each script, expected
directory structure, and the diagnostic plotting requirements.

Summary of what needs to be built:

- **0a.** `scripts/sbi/train_jaxili_from_compressed.py` (~250 lines, NEW) —
  standalone jaxili NDE training on pre-compressed datasets. Consumes
  `cnn_train.npz`/`cnn_val.npz`/`cnn_obs.npz`, trains jaxili MAF, samples
  posteriors, computes all metrics.
- **0b.** MAF companion for VMIM training — modify `npe_cnn_nbody_tomo.py` to
  support `--vmim-companion-backend {sbi_lens,maf}`. Replace the 4-layer
  sbi_lens RealNVP companion with a distrax/Haiku MAF for arms 7-8, 10.
- **0c.** `scripts/sbi/run_definitive_comparison.py` (~400 lines, NEW) —
  campaign orchestrator (Phase A compressors → Phase A.5 compression →
  Phase B NDE → Phase C analysis).
- **0d.** Metric computation utility — FoM3 + 2D FoM per parameter pair +
  marginal σ for all 6 parameters, saved to `.fom.json`.
- **0e.** Per-run diagnostic plotting suite — compressor loss curves, feature
  distributions, NDE val loss, corner plots, live dashboard. Details in plan.

## Connections

- Triggered by: `[[comprehensive-experiment-audit-2026-05]]`
- Predecessors: `[[canonical-anchors-refresh-2026-05]]`, `[[cnn-auto-push-18-20-2026]]`
- **Implementation plan**: `~/.claude/plans/mighty-tumbling-sparrow.md` — READ THIS before any code work
