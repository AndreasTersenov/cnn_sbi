# HANDOFF — flat-sky L1-vs-CNN + BNT (start here)

**Date:** 2026-06-11 (~13:30 UTC). **For:** the next Claude Code session (Fable 5).
**Branch:** `autoresearch/cnn-auto-push-18-20-2026` (continue on it; pushed through `b551586`).
**Felt fiber:** `.felt/flatsky-cross-2026-06/flatsky-cross-2026-06.md` — READ the top stanzas AND
keep prepending stanzas as you work (newest at top; commit the .md by path, never `.felt/index.db`).
**Conda env:** `jaxili` (`conda run -n jaxili python …` or `/home/tersenov/anaconda3/envs/jaxili/bin/python`).
**Previous handoff:** `HANDOFF_FABLE5_2026-06-10.md` (background; superseded by this one).

---

## 1. One-paragraph state

Both paper pillars are DONE, calibrated, figured, and drafted. **Pillar 1** (de-leaked flat-sky
cross): L1 gains +20% from the explicit product cross-map; the CNN's cross effect is zero ±
compressor-seed noise (sign flips with the draw); every CNN product seed ≤ 0.85× L1 product —
robust to seed (multiseed check) AND recipe (160k + de-noised best-val check: nothing moves);
auto-only is a statistical tie. **Pillar 2** (BNT): all three predictions HOLD — L1-auto inflates
to 0.15× FoM3 (σ_s8 doubles), L1+product partially recovers (0.22×), the CNN is near-lossless
(0.93×/0.88× over 3 compressor seeds, marginals ≤3%) ⇒ BNT inflation is an analysis-basis
artifact of per-channel statistics; GATE C on the BNT arms = pass with caveats, headline-safe
(L1-BNT mildly over-confident ⇒ inflation claims conservative; CNN-product L-C2ST 40% ⇒
headline losslessness from the AUTO arm). A **WHITENING TEST is RUNNING right now** (§4.1) to
decompose the L1 collapse into noise-geometry vs irreducibly-joint components. The other live
thread is **deepening the theoretical writeup** (§4.2) — Andreas explicitly found the last
explanation not thorough enough.

## 2. Read these first, in order

1. This doc + the felt fiber's top 3 stanzas (live state).
2. `FLATSKY_BNT_RESULT.md` (root) — pillar-2 numbers, marginals, GATE-C caveats, figure inventory.
3. `FLATSKY_CNN_RESULT.md` (root) — pillar-1 numbers incl. the multiseed + recipe robustness
   sections (all generator-derived from artifacts).
4. `scripts/sbi/results/exploratory/flatsky_cross_2026_06/PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md`
   — the paper draft: Part I (cosmologist main text), Part II (formal appendix, post red-team),
   Part III (joint-PDF / Cramér–Wold / Martinet), Part IV (point-cloud picture). **§4.2 below.**
5. Memory index `…/memory/MEMORY.md` — esp. `project_flatsky_bnt_losslessness`,
   `project_flatsky_cnn_no_cross_gain` (multiseed+recipe-refined), `feedback_gpu1_only`
   (UPDATED: pool is 0/1/2 now), `feedback_benchmark_dont_assume`, `feedback_no_pkill_self_match`.
6. `…/flatsky_cross_2026_06/bnt_campaign/gate_c/GATE_C_BNT.md` — the BNT calibration verdicts.
7. (When relevant) `EFFICIENCY_AUDIT_2026-06-10.md` + `PIPELINE_AUDIT_2026-06-10.md` — the audit
   docs; the remaining unimplemented item is Tier-1 scheduler packing (§4.3).

## 3. The numbers (pooled 3-MAF-seed, 9000-obs median FoM3; common jaxili MAF)

**Pillar 1 (no-BNT):** L1 auto 2405 | L1 +product 2875 (1.20×) | CNN auto 2325/2170/2480
(3 compressor seeds; straddles L1) | CNN +product 2181/2393/2433 (product/auto = 0.94/1.10/0.98
— zero systematic gain ± 8% seed noise) | CNN/L1 product 0.76–0.85×, robust to seed AND to the
160k recipe (lift auto 1.02×, product 1.00×). NEVER quote single-compressor-seed cross-gains.

**Pillar 2 (BNT/noBNT inflation):** L1 auto **0.15×** (2405→364; σ_s8 0.082→0.176) | L1 +product
**0.22×** (637) | CNN auto **0.93×** (0.94/1.00/0.86) | CNN +product **0.88×** (marginals ≤3%).
Plain CNN sufficed (the 20° campaign's advanced-arch contingency was NOT needed at 10°).
GATE C BNT: SBC — CNN mildly conservative (std 0.273–0.283), L1 mildly over-confident
(0.295–0.304 ⇒ true inflation ≥ measured ⇒ claims conservative); TARP dim-3 — global (un-split)
curves on the diagonal for all arms, tercile-resolved mild structure (cnn-auto HIGH −0.068);
L-C2ST — cnn-auto 13% (mild, self-test powered), cnn-product 40% (locally miscalibrated ⇒
headline from the AUTO arm).

## 4. LIVE THREADS (priority order)

### 4.1 WHITENING TEST — RUNNING (do not relaunch; check, interpret, write up)

`run_flatsky_whiten_campaign.py`, launched ~12:45 UTC detached on **GPU 1** (driver pid 3108884).
Per-channel L1 in the basis Q = (BB^T)^(−1/2)B — orthogonal (verified 4e-8), so = noise-whitened
BNT = an orthogonal rotation of the ORIGINAL basis with independent equal-variance noise
restored. **Purpose:** decompose the L1 BNT collapse — `recovered fraction =
(whiten − BNT)/(noBNT − BNT)`; high ⇒ the inflation was dominantly noise-ellipsoid geometry;
low ⇒ dominantly irreducibly-joint information. DIAGNOSTIC ONLY (remixing destroys the nulled
kernels — not a practical recipe; see Part II framing correction).
- Phases: sigma freeze (--mode whiten; check GATE A1b in `whiten_campaign/logs/sigma_*.log` —
  per-bin amp factors should be ≈1.000) → L1 both-whiten build (solo, the ~50–60 min loader
  pass) → {none, product} slices → fiducial precompute → per-arm slice → 2 jit sweeps →
  **`…/flatsky_cross_2026_06/whiten_campaign/WHITEN_RESULT.md`** (derived verdict ladder
  >0.8 / 0.4–0.8 / <0.4 recovered).
- **The previous session's monitors are DEAD** — first action: check
  `whiten_campaign/driver.out` (phase log) and whether WHITEN_RESULT.md exists. Single-GPU run,
  ETA roughly 15:30–16:30 UTC; if a phase FAILed, read `whiten_campaign/logs/<phase>_*.log`.
- When done: fold the recovered fraction into `FLATSKY_BNT_RESULT.md` (a derived line — extend
  the figure/results inventory), Part II/IV of the paper doc (the decomposition was
  pre-registered there), the memory `project_flatsky_bnt_losslessness`, and the fiber. A
  whitened-vs-noBNT-vs-BNT corner overlay or bar addition to `fom3_bnt_inflation` would be the
  natural figure (reuse `bnt_campaign_figures.py` patterns).

### 4.2 THEORY DEEP-DIVE — Andreas explicitly wants MORE depth (his words: "I was expecting a
much more thorough answer")

Context: he asked (a) which cross-correlations exactly are lost under BNT / where the
information goes / whether other higher-order correlations are lost / what operations on auto
maps recover them; (b) how the joint PDF is defined, computed, and used realistically; (c) how
the Martinet union-catalog approach maps onto the auto-map approach and whether it is complete.
The session answered with the "point-cloud/shadows" picture (Part IV of the paper doc), the
key sharp results being: BNT moves information ONLY within-pixel/equal-scale (pixelwise
transform) ⇒ per-scale joint one-point PDF provably suffices for BNT recovery; union-catalog
maps = count-weighted linear combos of the auto maps (catalogs add NO field information);
Cramér–Wold = the completeness theorem (union maps = finite Radon sampling of the joint PDF);
joint PDF is SBI-practical (no covariance obstacle; ~10^3-cell histograms ≈ current datavector
dims). **What's missing / what to do:** a genuinely thorough, self-contained treatment — e.g.
(i) work the 2-bin toy fully and honestly (Gaussian Fisher for diag-only vs full covariance,
the estimator-inefficiency vs information-loss distinction, with explicit formulas — the prior
session caught that naive toys can show NO asymptotic Gaussian loss, so the real-case statement
needs care); (ii) the σ8-anisotropy of the damage (σ8 worst, w0 mildest — dumbbell figure) tied
to the mechanism; (iii) possibly small numerical demos (e.g. a 2-bin Gaussian simulation
showing diag-vs-joint Fisher); (iv) consolidate Parts I–IV into one coherent, layered document
rather than accreted parts. Interview Andreas about which aspects felt thin before writing.

### 4.3 Backlog (in rough priority)
- **Tier-1 scheduler packing** (EFFICIENCY_AUDIT): multi-slot greedy schedulers + tenant probe +
  measured packing defaults. The gating benchmarks (sweep 3/GPU same-day, compressor 2/GPU,
  cross-class) are specified in the audit doc, never run. Sweeps are now jit-fast (~30 min/arm)
  so the payoff case is future campaigns.
- **Paper assembly** — all results/drafts/figures exist; `/paper-draft` when Andreas calls it.
  Remember `project_paper_narrative_includes_journey` (the paper must showcase the journey).
- **Joint-PDF third-pillar idea** (Part III/IV): wavelet-domain joint or pairwise-2D histogram
  datavector through the existing pipeline — would be NEW and is implementable with current
  machinery. Needs Andreas's explicit go (it's a new campaign).
- GATE C BNT cnn-product 40% L-C2ST — carried as a caveat; investigate only if Andreas asks.

## 5. New machinery this session (all committed; cf. fiber stanzas for details)

- **BNT wiring:** `flatsky_cross.apply_bnt_{np,torch,jax}` + `bnt=`/mode switch on all three
  `build_channels_*`; CNN `--flatsky-bnt`; L1 `--apply-bnt` or `--flatsky-channel-mix
  {none,bnt,whiten}` on flat_local; sigma tables carry a `mode` key HARD-ENFORCED at
  `select_frozen_sigma` (wrong-table = error, not warning). GATE A extended (8 op×mode combos).
- **Whiten mode:** `whiten_matrix_np()` (Q orthogonal); same plumbing end-to-end.
- **Campaign drivers** (all phase-barriered, SKIP-on-cmd-build-failure, derived verdicts):
  `run_flatsky_bnt_campaign.py` (done), `run_bnt_gate_c.py` (done; corners-first ordering),
  `run_flatsky_whiten_campaign.py` (RUNNING), `run_multiseed_compressor_check.py` +
  `run_recipe_160k_check.py` (done).
- **Figures:** `bnt_campaign_figures.py` (inflation bars, σ dumbbells, SBC grid, L-C2ST),
  `bnt_corner_overlays.py` (4 BNT-vs-noBNT corners), `plot_tarp_bnt_colored.py` (campaign
  colors + bands; `make_full` = un-split 600-pt version), `plot_bnt_datavectors.py` (σ8-coded
  datavectors: absolute, relative, and the `l1_hist_vs_s8_viridis` grid twin). All committed
  under `bnt_campaign/figures/`.
- **Perf (2026-06-10):** population sweep sampling is JITTED by default (174×/call measured;
  full-arm validated FoM3 −0.39%; `--sample-eager` = legacy bit-exact). **Reproducibility
  contract for sweeps is now keys-not-bits** (TF32-level sample differences).
- **Audit fixes:** derived verdicts everywhere (the hardcoded-verdict bug was demonstrated live
  by the in-flight multiseed run), dead NaN guard fixed, thread caps in all entry points,
  `--compressor-val-batches` (de-noised best_val; 160k check used 16), channel_scale +
  effective-checkpoint-policy persisted in cache meta, PRNG-collision assert, GPU-3 default
  removed from repr-corners runner.

## 6. Guardrails & conventions (updated this session — some SUPERSEDE CLAUDE.md text)

- **GPU policy (2026-06-10, supersedes "GPU 1 only"):** pool = GPUs **0, 1, 2**, tenant-checked
  with `nvidia-smi` before EVERY launch (tenants come and go within minutes — observed); never
  squeeze beside an ACTIVE tenant; **GPU 3 never**; ≤50 CPU workers; pack dispatch-bound jobs
  when measured. titan has no scheduler — `setsid nohup … &`, poll with `pgrep -f "[b]racket"`.
- **git:** stage by path; never `add .`; never commit `.npz`/`.pkl`/`index.db` (safety-grep
  before each results commit); figures png/pdf + lightweight JSON OK. Pre-existing dirty files
  (notebooks, other handoffs, `gc.log` warning) are NOT ours — leave them.
- **Claims:** lead with pooled 9000-obs medians; multi-compressor-seed for any CNN cross claim;
  FoM3 may headline but report σ/2D; never trust a contour before GATE C; never PCA the L1
  datavector; derived verdicts only — never write a conclusion into a generator.
- **CWD discipline (bit this session twice):** the Bash session cwd drifts between repo root
  and `scripts/sbi` — `cat >>`/`git add` with the wrong relative path created a stray root file
  once (caught, removed) and failed staging twice. Use absolute paths or check `pwd` first.
- Plans to markdown + sign-off before building; GPU campaigns need Andreas's explicit go;
  report-only for audits until he confirms.

## 7. Quick environment recap

Data/TFDS/fiducial-cache paths and the three-layer pipeline architecture are in the project
`CLAUDE.md` (auto-loaded) and `HANDOFF_FABLE5_2026-06-10.md` §7/§7b. The L1 stack needs
`wl_stats_torch`; the CNN does not. Obs conventions: typical = perm16/patch23, favorable =
perm0/patch90; population = perm<50 × 180 = 9000. θ = [Ωm, σ8, w0, h0, ns, Ωb], h0/100.
