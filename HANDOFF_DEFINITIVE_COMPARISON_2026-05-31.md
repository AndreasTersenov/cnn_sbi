# HANDOFF — Definitive L1-vs-CNN comparison (2026-05-31, end of session)

**Read this first**, then `CLAUDE.md`, then the felt fiber
`.felt/definitive-l1-vs-cnn-2026-05/definitive-l1-vs-cnn-2026-05.md`. This session
ran the full definitive comparison + the companion sub-investigation + TARP coverage +
steps 1–4. Everything below is verified (jobs finished, files on disk), not guessed.

---

## 0. How to work on this project (carry these over)

- **Check, don't guess. Never fabricate.** Every number here was read off disk / a log.
  If you haven't measured a perf/time number on this node, say "I don't have a number yet."
  (Memories: `feedback_benchmark_dont_assume`, `feedback_dont_guess_time_estimates`.)
- **Plan before non-trivial code; get sign-off.** Andreas's "don't start coding" is load-bearing.
- **GPUs:** the project `CLAUDE.md` says "GPU 1 only", but Andreas **overrode that for this
  campaign to GPU 0 + GPU 1** (the L1 campaign that owned GPU 1 is finished). GPU 2/3 have
  other tenants — don't touch. Pin every job with `--cuda-visible-devices`.
- **Git:** never `git add .`/`-A`; stage by path. **Don't commit without explicit OK.**
  Tree is chronically dirty — that's normal.
- **Felt (use it properly — see §6):** felt is a *persistent-context research substrate* (a tree of
  markdown "fibers" + a CLI), **not** an autonomous-loop driver. Before any felt work, **load the
  canonical skill** `~/.claude/plugins/marketplaces/cailmdaley-felt/claude-plugin/skills/felt/SKILL.md`,
  then read `FELT_AGENT_GUIDE.md` + `FELT_TUTORIAL.md` + CLAUDE.md §"Felt / Ralph operating conventions".
  **Drive it with the CLI** (`felt ls` / `felt show` / `felt add` / `felt edit --status closed
  --outcome "…"` / `felt history`) as concerns crystallize — don't hand-edit frontmatter (that bypasses
  the event log + index). `[[wikilinks]]` are for **fibers only** (memories/docs in plain prose).
  Scope-check first: don't open fibers for bug-hunting / one-PR / pure hyperparameter sweeps.
- **Conda/env:** `conda run -n jaxili ...` OR directly `/home/tersenov/anaconda3/envs/jaxili/bin/python`
  (the latter is more reliable; `conda run` buffers stdout — add `--no-capture-output`/`PYTHONUNBUFFERED=1`).
  **Never install packages** (e.g. apache_beam) — breaks the TF/protobuf stack.
- **Memories live in** `~/.claude/projects/-mnt-home-tersenov-software-cnn-sbi/memory/`
  (indexed in `MEMORY.md`). I updated several this session — read them.

---

## 1. Where the campaign stands (DONE)

The "definitive L1-vs-CNN" comparison is **substantively complete**: 10 arms, jaxili MAF NDE
for all, σ/2D primary (FoM3 secondary), TARP coverage for all, Phase C table written.

**All artifacts under** `scripts/sbi/results/exploratory/definitive_comparison/` (call it `$DC`):
- `PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md` + `phase_c.csv` — **the deliverable table** (10 arms).
- `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png` + `tarp_summary.json` — coverage,
  3 seeds (bands), all 6 single-perm arms.
- `companion_comparison_2026_05_31/` — MAF-vs-RealNVP (+ L1) overlays + table.
- Posteriors: `phaseB_tfdata_2026_05_30/` (CNN-RealNVP), `phaseB_maf_2026_05_31/` (CNN-MAF),
  `phaseB_std_2026_05_31/` (standardization), `phaseB_nativeauto_2026_05_31/` (native-TFDS auto),
  `phaseB_multiperm_2026_05_31/` (CNN perms 0/1/2), `posteriors/l1_*_split70/` (L1).
- Compressed caches (summaries): `phaseA_tfdata_2026_05_30/compressed/{autocross,autoonly}_s41/`,
  `phaseA_maf_2026_05_31/compressed/...`, `compressed/l1_*_dv/`.

---

## 2. Scientific findings (the results)

### Marginal σ (perm-0, 3 seeds pooled — the trustworthy metric; FoM3 is fragile)
| arm | σ(Ωm) | σ(σ8) | σ(w0) | FoM3 |
|---|---|---|---|---|
| L1 auto+cross | 0.027 | 0.042 | **0.125** | 34607 |
| CNN-RealNVP auto+cross | 0.027 | 0.038 | 0.151 | 26748 |
| CNN-RealNVP auto+cross (std) | 0.026 | 0.039 | 0.145 | 24281 |
| CNN-MAF auto+cross | 0.035 | 0.042 | 0.213 | 11984 |
| L1 auto-only | 0.039 | 0.052 | 0.204 | 10560 |
| **CNN auto native-TFDS** | 0.030 | 0.040 | **0.148** | **14969** |
| CNN-RealNVP auto-only (harmonic) | 0.035 | 0.042 | 0.216 | 9125 |
| CNN-MAF auto-only | 0.043 | 0.059 | 0.217 | 6679 |

### The findings
1. **PATCH-CENTER CONFOUND (G8) IS REAL AND LARGE — the headline.** native-TFDS auto-only is
   far tighter than the harmonic-cache-sliced auto-only (FoM3 14969 vs 9125, σ(w0) 0.148 vs 0.216).
   So the harmonic route's auto-only baseline is **lossy**. ⟹ the CNN cross-gain we quote
   (auto+cross / harmonic-auto-only = 2.93×) is **inflated by a poor baseline**; over a *fair*
   auto-only it's **~1.8×**. The within-route cross-channel effect is still valid (only channels
   differ), but the **magnitude is route-sensitive** — must be stated in any writeup.
2. **MAF companion is WORSE than RealNVP** (auto+cross FoM3 ~0.45× at cs41; ≤ RealNVP across all 5
   seed pairings; σ uniformly wider). ⟹ companion flow quality is **not** the CNN bottleneck.
   Sub-investigation **CLOSED**. (Classic "companion log-prob ≠ FoM3": MAF had *lower* VMIM loss.)
3. **Standardization is ~neutral** (σ marginally tighter, FoM3 marginally lower) — does NOT
   destroy information here. Plan arm 6 answered.
4. **L1 ≥ CNN-RealNVP** on auto+cross, driven by **w₀** (σ 0.125 vs 0.151); comparable on Ωm/σ8.
   (NDE-swap memory already said L1's edge is concentrated in w₀ — consistent.)
5. **TARP: all arms reasonably calibrated** (mildly over-confident mid-range, none severe; L1
   auto-only the most calibrated/slightly conservative). The tight contours are trustworthy.
6. **Leakage is empirically negligible** (Andreas's prior tests; dataset big/expressive). The
   "fast tf.data route leaks ~1.6×" alarm is **overstated** — treat fast-route absolute FoM as fine.
   Clean disjoint rerun **DEPRIORITIZED** (and blocked anyway: `.npz` loader is GIL-bound).

---

## 3. Open items / next steps (nothing is mid-flight; pick up cleanly)

1. **Per-perm-averaged multi-perm comparison.** `aggregate_all_arms.py` currently **pools all 3
   perms** for the multi-perm arms (autocross 7868 / auto-only 6124) — broader than the perm-0
   arms, **NOT apples-to-apples**. Fix: compute FoM3/σ **per perm then average** for multi-perm,
   and/or pool L1 over its 3 perms too. Then the perm-matched L1-vs-CNN comparison is fair.
2. **Write the step-3 patch-center confound prominently into `SUMMARY_DEFINITIVE.md`** — it
   changes how the cross-map gain should be quoted (~1.8×, not 2.9×).
3. **TARP for the new arms** (std, native-TFDS auto, multi-perm) — only the 6 original arms have
   TARP. Use `tarp_from_compressed.py` → dump → `run_tarp_coverage.py` (see §4).
4. **Commit the session's code** (with Andreas's OK) — see §5 for the uncommitted list.
5. **Deeper follow-ups Andreas is steering (NOT auto-queue):** 120k compressor steps (plan's own
   follow-up; deeper CNNs overfit at 120k per `project_resnet50gn_120k_overfits`); the **w₀
   question** (why L1 wins on w₀); SBC as a coverage cross-check. Ask before launching.

---

## 4. Infrastructure (scripts I added — how to reuse)

- **`tarp_from_compressed.py`** — trains the jaxili NDE from a compressed cache (cnn_*.npz OR
  l1_*.npz, auto-detected) + dumps `posterior_samples.npz` (N=200 val test points × M=2000) in the
  format `run_tarp_coverage.py` expects: `<dumps>/<arm>/seed_<S>/n200_m2000/posterior_samples.npz`
  with `samples (N,M,6)`, `theta (N,6)`. **Pass ABSOLUTE paths** (orbax requirement).
- **`run_tarp_coverage.py`** (pre-existing, I added arm colors) — consumes dumps → 3-D + 6-D
  curves, per-arm + overlay. Unknown arm names fall back to gray, so add to `ARM_COLOR`/`ARM_DISPLAY`.
- **`aggregate_all_arms.py`** — Phase C: discovers all arm posteriors → σ/2D/FoM3 table →
  `PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md`. Registry `ARMS` (dict of label→glob); add new arms there.
- **`run_steps_overnight_2026_05_31.sh`** — the steps-1–4 orchestrator (done; for reference).
- **`run_multiperm_fixed.sh`** — the working multi-perm runner (obs recompress + NDE).
- **`train_jaxili_from_compressed.py`** — the standalone jaxili NDE (also used for production
  posteriors). `--obs-files p0,p1,p2` does multi-perm; `--standardize-summary` does z-scoring.
- **`vmim_maf_companion.py`** + `test_vmim_maf_companion.py` — the MAF companion (committed) +
  its correctness tests (identity-init==N(0,I); log-det==autograd Jacobian). Flag:
  `--vmim-companion-backend maf` on `npe_cnn_nbody_tomo.py`.

**Multi-perm recompress (the working recipe — DON'T use `--no-train`):** to compress the fiducial
obs at perm p with a trained compressor, **omit `--train-compressor`** and pass
`--compressor-params <.../params_nd_compressor_best_val.pkl> --compressor-state <...opt_state_resnet_best_val.pkl>`
`--harmonic-obs-perm p --harmonic-train-realizations-limit 1 --harmonic-val-realizations-limit 1
--exit-after-compress` → grab `cnn_obs.npz`. (best_val pkl lives under
`<save-dir>/vmim/nbody/sigma_0.26/gal_density_30/bin_4/harmonic_nobnt_ch{10|4}/`.)

---

## 5. Gotchas I hit (so you don't repeat them)

- **orbax checkpoints require ABSOLUTE paths** — relative `--output-dir`/`--dumps-root` →
  `ValueError: Checkpoint path should be absolute`. (My scripts now `.resolve()` internally.)
- **Launch-redirect-before-mkdir:** `setsid nohup ./x.sh > $ROOT/nohup.out &` fails silently if
  `$ROOT` doesn't exist yet (script's own `mkdir` runs too late). **`mkdir -p $ROOT` before launching.**
- **`--no-train` is for the FLOW/NDE, not the compressor.** To load a trained compressor, omit
  `--train-compressor` + pass `--compressor-params/--compressor-state` (see §4). `--no-train` +
  `--save-dir` falls back to a default `tomo/save_params/...batch150000.pkl` path → FileNotFound.
- **`pgrep -f "name"` self-matches** if your own shell's argv contains "name" (e.g. a wait-loop
  whose command text includes the script name). Use the `[_]` bracket trick or kill by stored PID.
  (Memory `feedback_no_pkill_self_match`.)
- **`.npz` harmonic loader is GIL-bound** — `--harmonic-loader-threads 24` did NOT speed it up
  (2.25 it/s, same as 4 threads). A multiprocessing loader would be needed; deprioritized.
- **NEVER PCA the L1 datavector** (`--pca-components 0`); L1 cross MUST use the harmonic-cache
  route. (Memories `feedback_never_pca_l1`, `feedback_l1_cross_must_use_harmonic_route`.)

### Uncommitted code (this session) — recommend committing with Andreas's OK
- **Modified:** `scripts/sbi/npe_cnn_nbody_tomo.py` (additive: `--harmonic-loader-threads/pool/prefetch`
  passthrough — note it doesn't help, GIL-bound), `scripts/sbi/run_tarp_coverage.py` (additive arm colors).
- **New:** `tarp_from_compressed.py`, `aggregate_all_arms.py`, `run_multiperm_fixed.sh`,
  `run_steps_overnight_2026_05_31.sh`, `run_tarp_*.sh`, `run_cnn_phaseA/B_*.sh`, plus result dirs.
- **Committed already:** `0d58d5e` (MAF companion + tests).

---

## 6. Felt — current tree + how to use it (read before touching felt)

**Load order:** canonical skill (`~/.claude/plugins/marketplaces/cailmdaley-felt/claude-plugin/skills/felt/SKILL.md`)
→ `FELT_AGENT_GUIDE.md` → `FELT_TUTORIAL.md` → CLAUDE.md §"Felt / Ralph operating conventions" →
`felt ls` → the constitution. Felt is a research-note substrate; **drive it with the CLI**, don't
hand-edit `.md` frontmatter (the CLI keeps the append-only event log + FTS index consistent; bodies
*are* edited with a text editor).

**The campaign's tree is now felt-native** (`felt tree`):
- Constitution `definitive-l1-vs-cnn-2026-05` (open). Declared primary metric: 3-seed pooled FoM3
  on (Ωm,σ8,w0); σ/2D secondary. ⚠️ **metric drift this session** — we reasoned in σ/2D while the
  declared metric is FoM3. Either formally switch the declared metric or stick to it (don't mix).
- `…/finding-patch-center-confound-g8` (closed) — the G8 result + the "quote cross-gain ~1.8×" caveat.
- `…/maf-companion-not-bottleneck` (closed) — companion sub-investigation.
- `…/bug-multiperm-no-train-flag` (closed) — the bug + the fix recipe.
- **`…/refine-phase-c-perm-matched` (OPEN) — THIS IS YOUR NEXT TASK** (the §3 refinements).
  Start by reading it: `felt show definitive-l1-vs-cnn-2026-05/refine-phase-c-perm-matched`.

**Work it properly:** as concerns crystallize, `felt add <campaign>/<slug> "name" -t <tags> [-s closed]
-o "verdict + number + pointer + next"`; close with real outcomes (never "done"); `[[fiber-slug]]`
to link. When the campaign hits its done-condition, `felt edit <campaign> --status closed --outcome
"…"` and append an **Exit Interview** to the body (FELT_TUTORIAL §9). Keep `felt check` clean (it is
now — `[[wikilinks]]` resolve to fibers only; memories/docs go in plain prose).

**Honest note:** this session *under-used* the CLI — work was hand-driven, fibers filed at the end.
Do better: `felt ls` at the start, file fibers as you go, close them as evidence lands.
