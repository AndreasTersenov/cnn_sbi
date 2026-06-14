# PLAN — Phase 1: NDE-flow comparison on frozen CNN-VMIM summaries

**Status:** SIGNED OFF (Andreas green light 2026-06-13); Phase 0 DONE; B1 oracle launched.
**Session:** CNN-optimization (`.felt/cnn-optimization-2026-06`). **Entry point:**
`HANDOFF_CNN_OPTIMIZATION.md`. **Scope:** CNN side only. Do NOT touch L1/BNT/joint-stat work.

---

## Campaign goal & success criteria (referee-defense — Andreas 2026-06-13)

The paper compares analytical HOS (l1-norm) vs "optimal" CNN neural summaries on constraining
power. Current finding **l1 > CNN is surprising** (CNNs are *supposed* to be sufficient
statistics) and invites the attack "impossible — you trained the CNN suboptimally." Outcome
ladder — both of the first two are good outcomes:
1. **CNN > l1** — reclaims CNN optimality; resolves the surprise. Fine.
2. **CNN ≈ l1 (TIE)** — l1 happens to be a sufficient statistic. Clean, least-surprising. Fine.
3. **l1 > CNN persists** — defensible (NNs optimal only in theory) **only if** we show an
   exhaustive "tried every reasonable lever" record. That record is itself publishable as a
   **prescription for optimizing learned compressors**.

**Cross-channel framing (Andreas 2026-06-13):** the CNN sees all 4 tomographic auto-maps
jointly, so it already finds the useful cross-bin correlations *implicitly* — we do NOT expect
it to benefit from explicit hand-built cross channels (product/conv). So "+product hurts the
CNN" is EXPECTED (redundant input), not a defect. ⇒ The fair, headline comparison is
**CNN auto-only vs L1+product**: both then access the same cross-correlation information (the
CNN implicitly, L1 explicitly). The CNN arm of record is **auto-only**; do not feed it cross
channels. The real result is the CNN-auto (2325) → L1+product (2875) gap: the CNN is not yet
extracting all the cross-correlation power that is present in its joint auto-map input.

⇒ Success is measured by *comprehensiveness + defensibility*, not just the FoM3. A tie is a win.
Negative results are assets (which lever didn't help, and why). See memory
`project_cnn_optimization_goal_referee_defense`, `project_paper_narrative_includes_journey`.

**Skeptical-NN-referee lever catalog (must cover or justify skipping before concluding "CNN
can't do better"):** (a) **NDE flow** family + capacity ← *this plan, Phase 1*; (b) **architecture**
— ResNet/GroupNorm, depth/width, attention (Phase 2); (c) **VMIM objective + companion flow**
(Phase 3); (d) **training budget / schedule / early-stopping / LR**; (e) **summary dimension**;
(f) **data augmentation**; (g) **seed-robustness / initialization**. Each lever: report FoM3
effect + mechanism. The independent adjudicator guards each milestone against "you did it wrong."

---

## 0. The reframing that sets this plan

Andreas's lever-1 hypothesis was "jaxili MAF is a poor NDE for the CNN summaries; RealNVP did
better in the 20° analysis." **Reconnaissance finding:** the current flat-local CNN numbers
(auto-only 2620 etc.) were produced by `npe_cnn_nbody_tomo.py`, whose inference NDE is
**already sbi_lens `ConditionalRealNVP`** (`######## TRAINING FLOW ########`, line 3716;
`--nvp-layers 4 --nvp-hidden 128`), trained on the frozen VMIM summaries. The jaxili-MAF path
(`train_jaxili_from_compressed.py`) is what the 20° *definitive_comparison* used.

So production is already on the RealNVP family. That leaves two live forms of "the NDE caps
the CNN," and this phase tests both at once:
1. **Under-capacity / under-training of the current RealNVP** (4 affine-coupling layers, width
   128, is shallow for a 6-D posterior — a candidate cause of the seed-fragility 2620/2364/2387).
2. **A different family genuinely beats it** on these summaries (MAF / NSF / MDN).

## 1. Objective, metric, bar (unchanged from the fiber)

- **Primary metric:** 9000-obs **median FoM3** (`population_sweep_flatsky.py`; perm<50 × 180
  patches), reported with σ(Ωm, σ8, w0) alongside. Rank by FoM3, **not** val-loss.
- **Bar:** L1+product **2875** (gate-C clean). CNN must reach it, calibrated, to defend M1.
- **Calibration mandatory:** a tighter-but-miscalibrated posterior does not count
  (LANE_A cautionary tale).
- **Corrected current standing (Phase 0, 2026-06-13) — the metric ALREADY fixes the NDE.**
  `population_sweep_flatsky.py` runs *every* arm (L1 and CNN) through a **common jaxili MAF**
  (5 transforms, hidden [50,50]) at the 9000-obs pooled-3-seed median. On that apples-to-apples
  metric: L1 auto **2405** / +product **2875** / +both 2910; CNN auto **2325** / +product 2181
  / +both 2306 (explicit cross channels are redundant for the CNN — expected, not a defect). ⇒
  fair comparison = **CNN auto-only 2325 vs L1+product 2875** (−19%); both access the same
  cross-correlation info (CNN implicitly via the joint auto-maps, L1 explicitly). The handoff's "2620" is the **best single un-pooled seed** (`plot_best_seed.py`),
  NOT the primary metric. **Sharpened lever-1 question:** is CNN's 2325 depressed because the
  *small default MAF* under-serves its 10-D (mildly collinear, corr-cond ≈ 300) summary — and
  would a stronger NDE lift CNN *more than it lifts L1*?

## 2. Controlled-experiment design — one factor (the NDE), everything else frozen

**Frozen input (Phase 0, already on disk — no GPU):**
`results/exploratory/flatsky_cross_2026_06/cnn_phase/cnn_none_s41/cache/`
- `cnn_train.npz` = (323640, 10) summaries + θ; `cnn_val.npz` = (504000, 10); `cnn_obs.npz`.
- `cnn_cache_meta.npz` records compressor arch (`plain`, conv 64/128/256, dense 256, dim 10),
  params path + SHA256, split config, channel_mode, zero-mean flag. **Provenance is captured.**
- This is the **auto-only seed-41** compressor (best current arm; common-MAF 9000-median 2325).
- **Split is example-disjoint (verified):** compressor trains on perms 0–4, the frozen NDE data
  (`cnn_train.npz` = 899×180×**2 perms 5–6**) on perms 5–6; production log audited `overlap=[]`.
  So B1 is a FAIR perm-disjoint measurement, not a leaky upper bound. Residual = same (cosmo,
  patch) under different perm-augmentations (the known, Andreas-judged-negligible
  `project_tfdata_cross_route_leakage`); and it is CNN-specific (L1 has no compressor), so if it
  inflates anything it inflates the CNN — making "CNN < L1" conservative, not optimistic.
- Phase-0 checks — **DONE 2026-06-13, all clean:** (a) SHA256 of `cnn_cache_meta` ↔ params
  `.pkl` MATCHES, and the params file is `..._best_val.pkl` (best-val ckpt, not last-step);
  (b) summaries healthy — no NaN/dead dims, 899 distinct cosmologies, mild collinearity
  (corr-cond ≈ 301, ~2 near-degenerate dirs of 10); (c) VMIM training curve check still pending
  (`loss_compressor_{train,test}.npy` on disk) — verify val plateau before fan-out.

**Every NDE variant reads this exact cache.** Byte-identical summaries ⇒ any FoM3 difference is
the NDE, full stop.

### 2a. The NDE matrix (screen: 2 seeds 41/42; finalists: +seed 43 → 3)

**Sub-test 1A — RealNVP capacity ladder (sbi_lens family; my prime suspicion):**
| id | layers | hidden | note |
|----|--------|--------|------|
| A0 | 4 | 128 | the production sbi_lens-RealNVP config (end-to-end `npe_cnn` NDE) |
| A1 | 6 | 256 | |
| A2 | 8 | 256 | |
| A3 | 8 | 512 | |

**Sub-test 1B — family swap at matched-ish capacity (~8 transforms / 256):**
| id | family | impl |
|----|--------|------|
| B1 | MAF (default) | jaxili `ConditionalMAF` — **harness-validation reference: must reproduce 2325** (the existing common-metric baseline; NOT a leaky "oracle" — see split note) |
| B2 | RealNVP | jaxili `ConditionalRealNVP` (separates *framework/training-loop* from *family*) |
| B3 | NSF | distrax rational-quadratic-spline coupling (MDN as weaker fallback) |

7 configs. Screen with 2 seeds (41/42); finalists +seed 43.

**The back-pressure oracle is B1, not A0.** The existing 2325 baseline was produced by the
default jaxili MAF (via `population_sweep_flatsky.py`), so **B1 (jaxili MAF default) is the
harness-validation run: it MUST reproduce 2325 within noise** before any other number is
trusted. The sbi_lens-RealNVP ladder (A0–A3) and B2/B3 are *fresh* measurements against that
2325 baseline — A0 specifically asks whether the production end-to-end RealNVP (which scored a
best-single-seed 2620) beats the default MAF on the pooled 9000-median.

### 2b. Harness

Lowest-friction path: extend the `train_*_from_compressed.py` pattern into one
`train_nde_from_compressed.py` with `--nde-family {sbilens_realnvp, jaxili_maf,
jaxili_realnvp, jaxili_mdn, nsf}` + `--nde-layers/--nde-hidden`, reading `cnn_train.npz` and
exposing a `sample(y, n)` so the **same** `population_sweep_flatsky.py` computes FoM3 for every
family identically. jaxili families can reuse `train_jaxili_from_compressed.py` directly. I
will reuse `compute_fom3`/`fom2d`/`marginal_stats` verbatim — no reimplementation of the metric.

### 2c. Apples-to-apples completion (MANDATORY — added Phase 0)

The primary metric already fixes the NDE across L1 and CNN. So a better NDE only defends M1 if
it lifts CNN *relative to L1* — not if it lifts both. **The winning CNN NDE (and the default-MAF
B1) must therefore also be run on the L1+product summaries**, and the gap recomputed on that
*same* flow. The decision is `CNN-on-best-NDE` vs `L1+product-on-best-NDE`, never CNN-on-new-NDE
vs L1-on-old-MAF. (This is the PAPER_MESSAGES cross-cutting rule: fix one NDE, run both through
it.) L1+product frozen summaries live under `population_sweep/flat_product/` inputs — locate the
train-cache + fiducial-summaries npz when we reach this step.

## 3. Gating cadence (Andreas: proxy-every, full-gate-finalists)

- **Every variant:** 9000-obs median FoM3 + σ, **plus** a quick TARP-DRP coverage read (cheap
  proxy) to flag gross miscalibration early.
- **Finalists (top 2 by FoM3 that pass the proxy):** full **GATE C** = TARP-DRP (common-MAF,
  3 seeds, FoM3-tercile-stratified via `tarp_stratified_val.py`) **+ SBC**, run exactly as the
  L1 arms. A FoM3 win that fails GATE C is discarded (LANE_A rule).

## 4. Independent adjudicator (Andreas: at each milestone)

At the Phase-1 milestone (best NDE chosen, before any M1 claim), spawn a **fresh-context
subagent** given ONLY the artifacts (the winning variant's posterior `.npy`, `meta.json`, the
`train_nde_from_compressed.py` script, the gate outputs) — never my reasoning or expected
answer. Charter: independently recompute FoM3 from raw samples and confirm it matches; audit
the compressor↔NDE split for leakage/overlap; check GATE C numbers against registered
thresholds; hunt the known failure modes (last-step-not-best ckpt, val-loss-as-FoM3-proxy,
PCA-on-features, standardization mismatch). Returns PASS / FAIL / SUSPECT with reasons. Pinned
to a separate model, run in background. The adjudicator's verdict is logged in STATUS before
the result is accepted.

## 5. Decision rule / Phase-1 done condition

The decision is on the **gap to L1+product re-measured on the SAME winning NDE** (§2c), not the
raw CNN FoM3:
- **If** the best NDE makes **CNN ≥ L1+product on that same flow**, calibrated (GATE C) →
  candidate M1 = "CNN ≥ L1." Lock, adjudicate, write result, then decide whether Phase 2
  (architecture) can push further.
- **If** CNN stays below L1+product across every NDE (ladder A0→A3 plateaus AND no family
  B2/B3 closes the gap, with B1 reproducing 2325) → **the NDE is not the limiter.** Record
  cleanly and advance to Phase 2 (compressor architecture). This is itself a publishable
  sub-result: it rules out the "your flow is too weak" referee attack on the NDE axis and
  localizes the deficit to the compressor.

## 6. Compute plan, bottlenecks & lightening levers

**Where the time actually goes (from reading `population_sweep_flatsky.py`):** the script
**retrains the 3 NDE seeds in-process** (lines 89–96) and *then* samples — so one script = the
whole per-variant pipeline. The 9000-obs sampling is **already jitted to ~1 ms/obs** (≈30 s for
the full sweep) and posteriors use only **m_samples=2000**. ⇒ the **3× NDE retrain is the
dominant cost**, NOT the 9000 obs. (To be confirmed by the timed A0 run, not assumed.)

**Is 9000 obs overkill for screening? Yes — measured.** Bootstrapping the existing seed-41
9000-patch FoM3 (mean 2329, std 214 ≈ 9%): median-FoM3 SE is **29 @ n=100 (1.3%)**, **21 @
n=200 (0.9%)**, **8 @ n=1000 (0.4%)** — all far below the ~5–10% inter-variant gaps we rank by.
⇒ **screen at n_obs = 1000 (Andreas's call — cheap, ~0.4% SE), run 9000 only on the winner.**

**Lightening levers (Andreas-adjudicated 2026-06-13):**
1. ✅ **Pack 3–4 variants concurrently on GPU 2** (throughput; NDE is ~4–8 GB). 21 runs → ~6
   waves. Biggest win, zero rigor cost.
2. ❌ **Subsample NDE training data — REJECTED.** Andreas: keep the **full 323k** throughout.
   A smaller train set would leave a permanent "was that part of why we missed L1?" doubt,
   especially once larger compressors enter. Not worth the confound. (The lever was sound on
   information grounds — NDE is cosmology-limited — but the cost is rigor/peace-of-mind.)
3. ✅ **2 NDE seeds to screen, 3 for finalists** (keep ≥2 — seed-variance is a signal here).
4. ✅ **n_obs 9000 → 1000 for screening** (cheap, ~0.4% SE).
5. ✅ **Do NOT** cut m_samples below 2000 (covariance needs it) or truncate NDE training short
   of convergence (early stopping handles it; truncating confounds the family comparison —
   you'd rank by convergence speed, not quality).

**Screening config** (variants A1–A3, B1–B3): `--n-obs 1000`, 2 seeds, **full 323k train**,
full-convergence early stopping, packed on GPU 2. **Final config** (top-2 finalists):
`--n-obs 9000`, 3 seeds, full 323k train, + full GATE C.

**A0 anchor run = the calibration step (full settings, timed):** reproduces ~2620 (harness
oracle) AND measures the real NDE-train/sample cost split AND, paired with a screening-config
A0, confirms the cheap screen reproduces full-A0 FoM3 within noise before I trust it on B/A.

**GPU discipline:** GPU 2 only (foreign tenant ~3.6 GB, 0% util — co-reside politely). Fresh
`nvidia-smi` before every launch; per-job `XLA_PYTHON_CLIENT_MEM_FRACTION ≈ 0.9/N`,
`PREALLOCATE=false`; ≤50 CPU workers. No scheduler → run detached, re-arm my own monitor to
poll `.done` markers. If a heavy foreign tenant appears on GPU 2, pause and surface — do NOT
spill to 0/1/3.

## 7. Artifacts, logging, back-pressure

Per variant: posterior `.npy` (or per-patch metrics `.npz`), `meta.json` (config fingerprint +
input-cache SHA256), FoM3 + σ, NDE + (reused) compressor training curves, proxy-TARP read.
Results under `results/exploratory/flatsky_cross_2026_06/cnn_phase/nde_sweep_2026_06_13/`.
**Felt:** prepend one stanza per substantive event to
`.felt/cnn-optimization-2026-06/cnn-optimization-2026-06.md` (commit the `.md` by path, never
`.felt/index.db`). STATUS headline number = the primary metric (9000-obs median FoM3), no
metric-mixing.

**How I'll know it worked (oracles):** (1) A0 reproduces ~2620; (2) FoM3 monotone-ish in
capacity until plateau; (3) GATE C clean on the finalist; (4) the independent adjudicator
returns PASS having recomputed FoM3 from raw samples.

## 8. What this phase will NOT do

No architecture changes (Phase 2). No VMIM-companion changes (Phase 3). No new sims / more
cosmologies (data is ample). No L1/BNT/joint work. No PCA on features. No ranking by val-loss.
No last-step checkpoints.
