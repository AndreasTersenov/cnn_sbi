# HANDOFF — flat-sky L1-vs-CNN + BNT theory + joint statistics (start here)

**Date:** 2026-06-12. **For:** the next Claude Code session (Fable 5).
**Branch:** `autoresearch/cnn-auto-push-18-20-2026` (continue on it; pushed through `11720eb`
plus this handoff's commit). **Supersedes:** `HANDOFF_FABLE5_2026-06-11.md` (historical).
**Felt fiber:** `.felt/flatsky-cross-2026-06/flatsky-cross-2026-06.md` — read the top 3
stanzas; keep prepending (newest at top; commit the .md by path, NEVER `.felt/index.db`).
**Conda env:** `jaxili` (`conda run -n jaxili python …` or
`/home/tersenov/anaconda3/envs/jaxili/bin/python`). **No jobs are running.** GPUs free at
last check (pool 0/1/2, tenant-check before EVERY launch, GPU 3 never).

---

## 1. One-paragraph state

Everything is DONE and calibrated except where flagged. **Pillar 1** (de-leaked flat-sky
cross): L1 +20% from the product map, CNN zero-within-seed-noise, auto-only tie — complete,
GATE-C'd, figured. **Pillar 2** (BNT): L1-auto collapses to 0.15×, CNN lossless 0.93×; the
**whitening test recovered FULLY (1.06/1.01)** ⇒ the collapse is a FRAME artifact (no
irreducibly-joint share); the **§5.4 deep-channel ladder** measured +1 deep → 0.730, +2 deep
→ 1.082 (saturation at ~2 depth-distinct deep directions). The **theory deep-dive**
(`BNT_THEORY_DEEP_DIVE.md` v2.1, in `scripts/sbi/results/exploratory/flatsky_cross_2026_06/`)
is the canonical treatment — proofs P1–P7 + P4c, closed-form Fisher toy (the F3 "trap"),
claims ledger, all Andreas-reviewed. The **overnight menu (2026-06-12)** measured the rescue
menu and the NEW joint one-point statistics: pairwise joint PDF / joint wavelet l1 reach the
l1+product level from AUTO maps alone (marginals equal-or-better: σ_s8 0.072 vs 0.075; FoM3
2794/2788 vs 2875); Gaussian share of the l1's BNT loss = 0.38; unions6 rescue = 1.178;
and the **grid-transport result** (P4c): binned joint estimators are NOT BNT-invariant
(fixed grid 0.45 → axis-adapted 0.70 → the shear is unimplementable axis-aligned — only the
CNN's first layer does it). **The joint-stat arms have NO calibration (GATE C pending)** —
that is the main gate before any of them enters the paper.

## 2. Read these, in order

1. This doc + the felt fiber's top 3 stanzas.
2. `scripts/sbi/results/exploratory/flatsky_cross_2026_06/BNT_THEORY_DEEP_DIVE.md` (v2.1) —
   THE theory canon: §0 claims ledger, §1 plain-language, §2 proofs (P4c = grid transport),
   §3 Fisher toy, §4.3 joint-stat results + adjudicated predictions, §5 post-mortem chain +
   §5.4 ladder + §5.6 synthesis. Andreas reviewed v1 ("not detailed, nothing proved") and
   v2 was rebuilt to his spec: single-scale, layered, derivations not assertions.
3. `scripts/sbi/results/exploratory/flatsky_cross_2026_06/overnight_menu/OVERNIGHT_RESULT.md`
   (+ `HANDOFF_OVERNIGHT_2026-06-12.md`) — the overnight tables, addenda, night synthesis.
4. `FLATSKY_BNT_RESULT.md` + `FLATSKY_CNN_RESULT.md` (root) — pillar records, now incl.
   whitening + ladder + overnight stanzas + full figure inventory.
5. Memory index — esp. `project_joint_onepoint_stats_and_grid_transport` (NEW),
   `project_flatsky_bnt_losslessness` (updated through the ladder), `feedback_*` hard rules.
6. `PLAN_OVERNIGHT_MENU.md`, `PLAN_BNTDEEP_TEST.md`, `PLAN_PACKING_BENCHMARKS.md` (same
   campaign dir) — every registered prediction/band and the packing decisions.

## 3. The numbers (full rigor = pooled 3-MAF-seed, 9000-obs medians)

**Baselines:** l1 auto 2405 (σ_s8 .082, σ_w0 .245) | l1 auto+product 2875 (.075/.238) |
l1 BNT auto 364 (.176/.323) | CNN: see FLATSKY_CNN_RESULT.md (pillar 1 unchanged).

**Rescue ladder (recovered = (arm−364)/(2405−364)):** +cov50 (P7, Gaussian sector) 0.38 |
+deep(avg) 0.730 | whitened (rotation Q) 1.06 | +deep2 (avg+κ₄) 1.082 | +unions6 1.178.
Reading: 62% of the l1's BNT loss is non-Gaussian; per-channel info saturates at ~2
depth-distinct deep directions; survey-practice unions fully rescue.

**Joint statistics (dequantized; the quotable arms have `q` suffix):**
pair2dq 2794/1460 (ratio 0.52) | jointl1q 2788/1517 (0.54) | full4dq K=4 2401/1078 (0.45)
| full4da (adaptive grid) 2085/1455 (0.70). Counts ≈ l1-weighting (info = joint occupancy;
the 2D heatmaps show the weighting just amplifies the joint-tail corners). Full-4D at K=4 =
exactly the l1-auto baseline (resolution beats joint order at fixed budget).

## 4. LIVE THREADS / NEXT ACTIONS (priority order; all need Andreas's go where marked)

1. **GATE C on the joint-stat arms** [needs go; ~2 h]: TARP+SBC on `pair2dq_nobnt` (and
   `jointl1q_nobnt` if quoted). Required before any joint-stat contour enters the paper —
   per-channel rule "never trust a contour before GATE C". The flat-local TARP/SBC pattern:
   `run_bnt_gate_c.py` (adapt; corners-first ordering). L-C2ST is underpowered at 3000-dim
   (memory `reference_lc2st_underpowered_highdim_l1`) — TARP+SBC suffice (Andreas accepted
   that for high-dim arms).
2. **Paper assembly** [Andreas's call]: `/paper-draft` — all results/drafts/figures exist.
   The paper drafts: `PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md` (Parts I–II, current with all
   results) + deep-dive as the appendix source. REMEMBER
   `project_paper_narrative_includes_journey`: the §5 falsified-prediction chain (F4 → sign
   structure → 0.730 → 1.082) and the grid-transport falsification ARE narrative assets.
   Open decisions Andreas flagged but hasn't ruled on: is joint-PDF-as-statistic a third
   pillar or a discussion section; quote the 38% Gaussian share in the BNT section; [REF]
   placeholders to fill (the null→cut→invert pipeline citation; the lossless-2pt-with-crosses
   report — Andreas relayed it, no citation pinned; do NOT fetch without asking).
3. **Cheap unrun follow-ups** [optional, ~30 min each, need go]: pair2da (decompose the
   pairwise 0.52 into grid-vs-incompleteness); pair2d K=20 + adaptive ranges in BNT basis
   (registered prediction band "0.52 < r < ~0.75, below the rotate-back ceiling"); K=15
   resolution scaling on noBNT.
4. **Tier-1 packing benchmarks** [deferred by decision]: run as the FIRST phase of the next
   real GPU campaign (PLAN_PACKING_BENCHMARKS.md: 3-pack only; footguns verified already
   fixed). Don't run standalone.

## 5. Machinery built this session (all committed; one-purpose scripts in `scripts/sbi/`)

- **Mix modes** in `flatsky_cross.py`: `deep` (1×4 avg), `deep2` (avg+e₄), `bnt_deep`,
  `unions6` (6 pair averages) + `n_built_channels()` (mode-aware counts; non-square mixes OK).
- **Concat arm builders**: `build_flatsky_bntdeep_arm.py` (--deep-mode; deep-block σ rows =
  √(M²·σ²_auto) exact; theta/perm/patch bit-equality HARD-ASSERTED against the parent BNT
  cache — NaN-batch skipping makes row order parameter-dependent, so loader params must
  mirror the parent build: train/perms 5-6/flip/seed 1001/batch 512; val/test/0-1/noflip/2001).
- **Joint statistics**: `flatsky_joint_stats.py` (cov50 / pair2d / full4d / jointl1
  reductions; SNR units, fixed [−5,5] clamp-to-edge or `--adaptive-ranges` percentile grids;
  `dequant_gen` = seeded U(0,1) — **MANDATORY for count features**, quasi-discrete sparse
  cells NaN the jaxili MAF, diagnosed 3×) + `build_flatsky_joint_arm.py` (standalone or
  --append-to with asserts).
- **Campaign drivers**: `run_flatsky_whiten_campaign.py` (done), `run_flatsky_bntdeep_campaign.py`
  (--variant deep/deep2; done), `run_flatsky_overnight_menu.py` (3-GPU slot workers, tenant
  politeness probe ≥12 GB back-off, screening 1-seed/3000-obs → auto-escalation re-sweeps
  3-seed/9000-obs; done), `run_full4d_retry.py` (--arms; dequantized rebuilds; done).
- **Figures**: `whiten_campaign_figure.py` (5-bar decomposition), `overnight_menu_figures.py`
  (joint-stats bars, invariance ratios, rescue ladder), `corner_overnight_joint.py`
  (--variant nobnt/bnt, --replot-only; retrains 3 seeds ~80 s each — checkpoint reload is
  BROKEN >1000-dim, memory `reference_jaxili_checkpoint_reload_truncation`; saved pooled
  samples in `overnight_menu/corners/*.npy`), `plot_joint_datavectors.py` (σ8-coded curve
  grids + native 2D heatmaps). All figures under `overnight_menu/figures/` +
  `whiten_campaign/figures/`; inventory in FLATSKY_BNT_RESULT.md.
- **Sweep recipe** (for any new arm): cache {l1_train,l1_val}.npz (theta,x) + fiducial npz
  (S,perm,patch,truth) → `population_sweep_flatsky.py` (log1p-zscore, clip 5, min-var 1e-5,
  NPE jaxili MAF, jitted sampling). Typical obs = perm16/patch23; favorable = perm0/patch90.

## 6. Guardrails & lessons (additions to CLAUDE.md / 06-11 handoff; those still apply)

- **Andreas's review bar for theory docs**: derivations not assertions; interview him on
  what felt thin BEFORE writing; no external fetches without asking (he declined a Martinet
  fetch — completeness claims stated generically); plain-language layer + formal layer;
  single-scale presentation (no wavelet indices in explanations); honest falsified
  predictions KEPT in the docs as journey material.
- **Claims style**: marginals-first reading (FoM3 fragility — pair2d "loses" 3% FoM3 to
  l1+product while every σ is equal-or-better); always carry the GATE-C status of an arm;
  registered predictions with verdict bands BEFORE data; derived verdicts only.
- **Bash discipline**: the session cwd DRIFTS (`cd` in compound commands persists) — two
  incidents this session; use absolute paths, `cd` back to repo root before git. NEVER
  `(cmd1 && cmd2 &)`-style backgrounding of a whole && list — `&` binds the entire list
  (one launch's tail looked in the wrong place because of this). Launch pattern that works:
  `(cd <dir> && setsid nohup <cmd> > <ABS-path>.out 2>&1 &)` then verify with
  `pgrep -f "[p]attern"` + absolute-path tail.
- **Monitors**: `Monitor` tool with driver.out diff + result-file existence + pgrep
  liveness; the previous session's monitors die with the session — re-arm your own.
- git: stage by path; never `.npz`/`.pkl`/`index.db` (safety-grep before results commits);
  pre-existing dirty files (notebooks, old handoffs, definitive_comparison, gc.log warning)
  are NOT ours — leave them.

## 7. Environment recap

Project CLAUDE.md (auto-loaded) has the pipeline architecture and dataset paths. The L1/joint
stack needs `wl_stats_torch`; frozen σ tables: `flatsky_cross_noise_sigma{,_bnt,_whiten}.npz`
(all GATE A1b-passed; mode key HARD-ENFORCED). θ = [Ωm, σ8, w0, h0, ns, Ωb], h0/100;
effective prior = CosmoGrid support, σ_prior ≈ (0.115, 0.288, 0.462) for (Ωm, σ8, w0).
Heavy caches under `overnight_menu/<arm>/cache/` and `bnt*_campaign/` are on disk,
reusable, NOT in git.
