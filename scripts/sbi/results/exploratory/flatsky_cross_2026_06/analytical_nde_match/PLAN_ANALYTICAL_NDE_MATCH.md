# PLAN / CONSTITUTION — Analytical-stats → best NDE → match the CNN (2026-06-14 overnight)

**Author:** Claude (overnight, autonomous). **Branch:** `analytical-nde-match-2026-06`.
**Env:** conda `jaxili` (`/home/tersenov/anaconda3/envs/jaxili/bin/python`). **GPUs:** pool 0/1/2,
GPU 3 NEVER; fresh `nvidia-smi` tenant check before every launch; foreign tenants present on
0/1/2 (≤4 GB each) — pack politely with `XLA_PYTHON_CLIENT_MEM_FRACTION` caps; ≤50 CPU workers.

## 0. The trigger (what changed tonight)
The CNN-optimization session fixed the CNN: **ResNet18 + sbi_lens RealNVP 4×128 → FoM3 3293.8**
(σ(Ωm,σ8,w0)=0.045,0.072,0.229), calibrated (fresh TARP+SBC, 2026-06-14). The NDE swap is what
did it (auto-only jaxili-MAF 2312 → sbi_lens RealNVP 3141; arch ResNet18 nudges to 3293). The best
calibrated **analytical** result is still **l1+product 2875** (gate-C clean) / flat_both 2910. So
the CNN now LEADS analytical by ~13–15%. Andreas: "back on the hunt — make the analytical stats as
close as possible to this best CNN result."

## 1. The scientific question (and why this is not circular)
Lane A (overnight_menu_2/LANE_A_CONCLUSION.md) already established the key fact: **FoM3 differences
of ~20–30% between methods are as likely to be NDE/estimation-path quality as physics.** Its own
*deeper recommendation*: "before ranking ANY statistics by FoM3, fix one NDE architecture +
training budget + convergence diagnostic and run every arm through it." Lane A could not finish
this because the good NDE was not yet identified and A1 used MAF (borderline calibration); the
`l1+product+VMIM` cell was never run.

**Now the good NDE is known** (sbi_lens RealNVP 4×128, the one that lifted the CNN AND is calibrated
— net-bias +0.039, conservative). So tonight executes lane-A's recommendation with the good NDE:

> Give every analytical representation the SAME best NDE the CNN uses (10-D VMIM compression →
> sbi_lens RealNVP 4×128), gate each cell identically (TARP+SBC), and read the PATTERN. How close
> does the best CALIBRATED analytical statistic get to the CNN's 3293?

The cleanest paper outcome is **analytical ≈ CNN, calibrated** ("the l1/joint statistic is
sufficient; the CNN's apparent lead was an NDE artifact"). A genuine, calibrated CNN > analytical
gap is also publishable (a real representation gap, with an exhaustive "we tried every NDE lever"
record — see `project_cnn_optimization_goal_referee_defense`).

## 2. Primary metric + the bar (ONE metric, per felt convention #1)
- **Primary metric:** median FoM3 over the common fiducial population (FoM3 = 1/√det C_3 over
  Ωm,σ8,w0), reported with σ(Ωm,σ8,w0) alongside (marginals-first). Screens use n_obs=1000;
  finalists n_obs=9000. Same fiducial obs sets, same preprocessing, same gate as every prior arm.
- **The bar:** CNN ResNet18+RealNVP **3293** (calibrated). Secondary reference: L1+product 2875 (the
  analytical baseline to beat / explain).
- **An arm "counts toward the goal" ONLY if it passes GATE C.** A tighter-but-miscalibrated
  posterior is fool's gold (the A1=3822 borderline cautionary tale, LANE_A_CONCLUSION.md). FoM3 may
  be headlined but always with σ + the fragility caveat.

## 3. GATE C (calibration; the keep-rule Andreas chose: "calibrated FoM3 + plateau-stop")
Run `tarp_stratified_val_nde.py` (NDE-family-aware, so it tests the ACTUAL flow under question) →
`run_tarp_coverage.py`. Verdict (from run_laneB_gate_c.py thresholds):
- worst-tercile |ECP−α| ≤ 0.05 AND SBC rank-std ∈ [0.275,0.305] → **PASS**
- worst |dev| ≤ 0.10 (and std band) → **PASS-with-caveat** (reportable, flagged)
- worst |dev| > 0.10 OR SBC std outside band → **FAIL** (tightness is over-confidence; the
  calibrated value is lower — quote the gated read, not the raw FoM3).
Net signed bias sign matters: + = conservative/over-covers (safe), − = over-confident (the failure
mode). Report net-bias per seed; require net-bias ≥ 0 (or ≤ small +) across compressor seeds for a
"win" claim (the registered condition A1 failed).

## 4. The experiment matrix (representation × NDE; every cell gated)
Representations (analytical caches that exist under `l1_matrix/` and `overnight_menu*/`):
- **l1-auto** (800-d) — control, no cross info. raw→MAF baseline 2405.
- **l1+product** (3200-d) — the trustworthy workhorse; raw→MAF 2875 **gate-CLEAN**. ξ_ij cross info.
- **pair2d** (joint 1-pt PDF of autos, ~K=10) — raw→MAF 2794 (gate FAIL, over-confident); the A1
  parent. Carries cross-bin info from autos alone.

NDE paths:
- **raw→jaxili MAF** — baselines (exist): l1-auto 2405, l1+product 2875 (clean), pair2d 2794 (FAIL).
- **VMIM(10-d)→jaxili MAF** — A1 pair2d = 3822/3441/3408 (borderline); l1+product+VMIM = **UNRUN
  (the 4th cell)**; l1-auto+VMIM = unrun.
- **VMIM(10-d)→sbi_lens RealNVP 4×128** — **ALL UNRUN; the new lever** matching the CNN exactly.
- (raw→sbi_lens RealNVP is known to crater high-D L1 to 1111 — documented negative, not re-run.)
- Optional secondary: VMIM→jaxili MDN 10×50 (the CNN sweep's #3 family, 2885).

VMIM compressor: `vmim_from_cache.py` (MLP 256,256 → 10-d, RealNVP companion 4×128, log1p-zscore
on the parent, steps 30000). 3 compressor seeds {41,42,43} on the headline arms (the A1 band showed
±5% seed spread — never quote a single compressor seed).

## 5. Registered predictions (branch sentences — written BEFORE the numbers; felt convention)
**pair2d-VMIM → RealNVP (reuse A1 caches s41/42/43):**
- FoM3 ≥ ~3100 (within ~6% of CNN) AND gate PASS / net-bias ≥0 across seeds → *"the analytical
  pairwise joint PDF, given the CNN's own NDE, MATCHES the CNN — the lead was an NDE-quality effect;
  analytical HOS are sufficient at this level."*
- FoM3 ≥ ~3100 but gate FAIL/net-bias<0 (like the MAF A1) → *"the compress-then-flow FoM3 gain does
  not survive calibration regardless of NDE family; the joint-PDF tightness is partly
  over-confidence (DPI artifact); the calibrated analytical ceiling is [gated value]."*
- FoM3 < ~3100 with RealNVP → *"RealNVP does not lift the compressed joint stat the way it lifts the
  CNN; the residual gap is representation-bound, not NDE-bound."*

**l1+product-VMIM → RealNVP (and →MAF, the unrun 4th cell):**
- ≥ ~3100 AND PASS → *"even the calibration-CLEAN l1+product, compressed + given the CNN's NDE,
  reaches the CNN — the strongest 'analytical = CNN' statement, on the trustworthy statistic."*
  (BEST outcome.)
- improves over 2875 but miscalibrates → artifact signature; clean ceiling stays ~2875.
- RealNVP ≈ MAF ≈ 2875 (no lift) → *"l1+product's MAF was already optimal; the CNN's ~15% edge over
  l1+product is REAL (representation), not NDE."*

**l1-auto-VMIM → {MAF,RealNVP}:** control; expected below the cross arms; if it ALSO jumps to ~CNN
that would mean the gain is pure NDE/compression on ANY summary (a DPI red flag to investigate).

**The deliverable is the PATTERN across the matrix, gated — not any single FoM3.**

## 6. Budget, plateau-stop, done condition (felt convention #2)
- **Budget:** ~8 h wall (Andreas asleep ~22:15→~06:30 UTC). The matrix is bounded (~9 core cells +
  gates) → completes within budget.
- **Plateau-stop:** stop launching NEW arms when 3 consecutive new arms land within ±5% of the
  running best CALIBRATED FoM3, OR the matrix is complete + gated, OR 06:30 UTC — whichever first.
- **Done condition / auto-close:** matrix complete + gated OR plateau OR 06:30 UTC. Then write
  RESULT doc + update memory + felt stanza + morning handoff regardless of outcome.

## 7. Execution phases
- **P0 validate (Andreas's standing rule):** smoke sbi_lens-on-compressed on existing A1 cache
  (n=200, 4k steps) — confirm FoM3 sane, not NaN, preproc chain correct (compressed meta says
  downstream preproc = none/0/1e-12 = the seam defaults). [LAUNCHED]
- **P1 (cheap, reuse A1 caches):** pair2d-VMIM → RealNVP, 3 compressor seeds, screen→full→gate.
- **P2 (long pole):** compress l1+product & l1-auto (3 / 1–3 seeds) [LAUNCHED s41 of each]; then
  run {MAF, RealNVP} on each compressed cache; screen→full→gate the survivors (≥~2875).
- **P3 synthesize:** the gated matrix table; plateau check; RESULT_ANALYTICAL_NDE_MATCH.md; memory;
  felt; HANDOFF.

## 8. Seam (all reusable; no core new code)
- `vmim_from_cache.py` — parent cache (l1_train/val.npz) + parent fiducial → 10-d compressed cache +
  compressed fiducial. Flags: `--cache-dir --fid-npz --out-cache --out-fid --summary-dim 10
  --seed --steps`.
- `train_nde_from_compressed.py` — compressed cache + `--nde-family {sbilens_realnvp,jaxili_maf,
  jaxili_mdn}` → median_summary.json (FoM3). Default preproc none/0/1e-12 (correct for compressed).
- `tarp_stratified_val_nde.py` + `run_tarp_coverage.py` — family-aware GATE C → dumps → terciles+SBC.
- Compressed caches use `--cache-prefix l1`; fiducial key `S` + `perm`/`patch`/`truth`.

## 9. Process discipline
Detached background jobs (harness-tracked; NO nohup+background double-tracking). Kill by stored PID
/ bracket trick, never `pkill -f` self-match. Stage commits by path; do NOT commit results/caches/
.npz/.pkl/index.db. NEVER PCA L1. Per-channel noise model for any cross/mixed channels (n/a here —
operating on pre-built caches).
