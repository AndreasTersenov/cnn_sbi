# HANDOFF — canonical-anchors-refresh campaign

**UPDATE 2026-05-26 ~06:30 UTC: campaign COMPUTE COMPLETE. All 12 canonical posteriors landed cleanly. See §"Final canonical anchors" below.** Remaining work is write-up only (METHODOLOGY.md, refresh of CNN_CROSS_MAPS_INFORMATION_NOTE.md, close fibers with exit interviews). No more compute needed.

**Read this first. Do not launch anything before completing the §"Before doing anything" checklist below.**

This document hands off the `canonical-anchors-refresh-2026-05` felt campaign to a fresh Claude session. The previous session got into a "launch → bug → relaunch" loop that consumed a lot of GPU. The new session should be more deliberate.

## Final canonical anchors (all four arms × 3 seeds, fully clean)

| arm | per-seed FoM3 | MoS | **Pool** | haircut | \|bias\| med |
|:---|:---|---:|---:|---:|---:|
| CNN auto-only | 18,060 / 17,732 / 14,845 | 16,879 | **12,873** | 0.76 | 0.48σ |
| CNN auto+cross | 19,699 / 14,914 / 18,214 | 17,609 | **12,615** | 0.72 | 0.19σ |
| L1 auto-only | 11,419 / 21,951 / 11,752 | 15,041 | **12,004** | 0.80 | 0.31σ |
| L1 auto+cross | 39,895 / 36,423 / 38,361 | 38,226 | **34,004** | 0.89 | 0.18σ |

**Verification**: L1 cross s43 came out at 38,361 — **identical to the v2_chsigma anchor**. Pool 34,004 vs stale 33,820 (within 0.5%). The canonical setup reproduces L1 cross exactly.

### Canonical ratios

| ratio | stale | **canonical** |
|:---|---:|---:|
| CNN cross/auto pool | 2.16× | **0.98×** — cross-maps essentially don't help CNN under clean splits |
| L1 cross/auto pool | 3.07× | **2.83×** — cross-maps give L1 a robust ~3× lift |
| CNN/L1 at auto-only | 1.23× | **1.07×** — essentially tied |
| CNN/L1 at auto+cross | 0.71× | **0.37×** — L1 dominates auto+cross by ~2.7× |

### Stale anchor shifts

| arm | stale → canonical | shift | interpretation |
|:---|---:|---:|:---|
| CNN auto-only | 11,130 → 12,873 | +16% | small clean-up from 70/30 splits |
| CNN auto+cross | 23,986 → 12,615 | **−47%** | the big contamination correction (train/train compressor-NDE overlap removed) |
| L1 auto-only | 11,073 → 12,004 | +8% | essentially unchanged |
| L1 auto+cross | 33,820 → 34,004 | +0.5% | matches v2_chsigma anchor exactly |

### Plots
- Bound PDF: `scripts/sbi/results/exploratory/canonical_anchors_refresh/canonical_diagnostics.pdf`
- PNGs: `canonical_diagnostics_png/00_overview/` (cross-arm comparisons) and `canonical_diagnostics_png/<arm>/seed_<N>/` (per-(arm, seed) detail pages)

---

---

## TL;DR

**Goal**: produce 4 canonical FoM3 anchors (CNN auto-only, CNN auto+cross, L1 auto-only, L1 auto+cross) on a uniform methodology, so the paper can quote them with confidence.

**Status (2026-05-26 ~05:46 UTC)**:
- ✅ 9 of 12 posteriors clean and on disk: CNN auto (3 seeds), CNN cross (3 seeds), L1 auto (3 seeds — PCA off, full train, matches stale anchors).
- 🟡 3 L1 auto+cross seeds running on GPU 1 right now (harmonic-cache route, channel-aware noise, PCA off — this is iteration 3 of L1 cross, the first two iterations had silent bugs).
- ❌ Arm 5 (CNN cache-auto-only sanity check) never re-launched after the bug fix; code is ready but no run has fired.

**Running compute** (do NOT kill — these are the correctly-configured runs):
- L1 cross s41: bash job `b5h20e82q`, PID 2678864
- L1 cross s42: bash job `bcgq3r2ck`, PID 2678972
- L1 cross s43: bash job `b4vzwosz2`, PID ~

**ETA**: ~30-50 min for all three to land.

---

## Before doing ANYTHING in the new session

1. **Read these memories** (in order):
   - `~/.claude/projects/-mnt-home-tersenov-software-cnn-sbi/memory/MEMORY.md` (index)
   - `feedback_never_pca_l1.md` — NEVER PCA the L1 datavector
   - `feedback_l1_cross_must_use_harmonic_route.md` — L1 cross MUST use `--full-sphere-cross-cache`
   - `feedback_gpu1_only.md` — only GPU 1 for new jobs
   - `feedback_val_loss_not_reliable_fom3_proxy.md` — val-loss ≠ FoM3 quality

2. **Read this campaign's fiber**:
   ```
   felt show canonical-anchors-refresh-2026-05
   ```
   Especially the "Loop Status (live)" stanza at the top of the body.

3. **Read the project CLAUDE.md** §"Felt / Ralph operating conventions" — the 7 mandatory rules.

4. **Run the pre-flight flag-diff tool BEFORE any compute**:
   ```
   conda run -n jaxili python scripts/sbi/results/exploratory/canonical_anchors_refresh/tools/flag_diff.py \
       <YOUR_LAUNCHER.sh> <ANCHOR.meta.json>
   ```
   It now has explicit gotcha checks for PCA and harmonic-route. Treat any RED finding as fatal.

5. **Check that the running L1 cross jobs are still alive** before deciding what to do:
   ```
   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv | grep -E "^[0-9]"
   ps -ef | grep "npe_l1norm_cross_jaxili" | grep -v grep | head
   ```

---

## Methodology — the 4 canonical arms

| arm | script | dataset / cache | NDE training data | epochs | key flags |
|:---|:---|:---|:---|:---|:---|
| CNN auto-only | `scripts/sbi/npe_cnn_nbody_tomo.py` | TFDS `grid_20deg_160px_nonoverlap48` | `train[70%:]` | 50k steps | `--zero-mean-maps --standardize-summary --compressor-checkpoint-policy best_val --require-disjoint-train-examples` |
| CNN auto+cross | same | harmonic cache `full_sphere_cache_grid` | `train[70%:]` | 50k steps | + `--cnn-map-route harmonic --full-sphere-cross-cache <path> --harmonic-cache-regime nobnt --harmonic-normalize-input-channels` |
| L1 auto-only | `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` | TFDS `grid_20deg_160px_nonoverlap48` | `train` (full 302k) | 5,000 epochs | `--zero-mean-maps --l1-min-snr -13 --l1-max-snr 13 --pca-components 0` |
| L1 auto+cross | same | **harmonic cache** (NOT TFDS+--cross-maps) | `train` (full) | 50,000 epochs | + `--full-sphere-cross-cache <path> --cross-noise-model channel_empirical_global --cross-snr-percentile 1.0 --pca-components 0` |

**Asymmetry**: CNN uses 70/30 split (compressor sees train[:70%], NDE sees disjoint train[70%:], with `--require-disjoint-train-examples`). L1 has no learned compressor so the 70/30 doesn't apply; L1 NDE trains on the **full** train. This is intentional — see `canonical-anchors-refresh-2026-05.md` Loop Status for rationale.

**Common to all 4 arms**: `--zero-mean-maps` (mandatory), `--cuda-visible-devices 1`, `--npe-samples 100000`, seeds 41/42/43.

The master launcher with all four correctly configured is at:
```
scripts/sbi/results/exploratory/canonical_anchors_refresh/launch_canonical_all.sh
```

---

## Anchors (canonical so far, L1 cross still landing)

| arm | per-seed FoM3 | MoS | Pool | haircut | \|bias\| med | notes |
|:---|:---|---:|---:|---:|---:|:---|
| CNN auto-only | 18,060 / 17,732 / 14,845 | 16,879 | **12,873** | 0.76 | 0.48σ | ✅ clean, all in posteriors/ |
| CNN auto+cross | 19,699 / 14,914 / 18,214 | 17,609 | **12,615** | 0.72 | 0.19σ | ✅ recovered via NDE-only re-run using saved best-val compressor (the 9h of compressor compute was preserved) |
| L1 auto-only | 11,419 / 21,951 / 11,752 | 15,041 | **12,004** | 0.80 | 0.31σ | ✅ matches stale anchor scale after PCA-off fix |
| L1 auto+cross | (running) | — | — | — | — | 🟡 third attempt — earlier ones had PCA-on then wrong route bugs |

**Comparison to stale (mixed-discipline) anchors**:
- CNN auto-only: 11,130 → 12,873 (+15.7%) ✅ small clean-up
- CNN auto+cross: 23,986 → 12,615 (−47%) — the proper 70/30 + disjoint-NDE correction
- L1 auto-only: 11,073 → 12,004 (+8.4%) ✅ matches
- L1 auto+cross: 33,820 → TBD when runs land

**Cross/auto ratios at canonical (so far)**:
- CNN cross/auto = 0.98× (stale was 2.16×)
- L1 cross/auto = TBD (stale was 3.07× on the corrected non-canonical setup)
- CNN/L1 at auto-only = 1.07×
- CNN/L1 at auto+cross = TBD

---

## What went wrong in the previous session (post-mortem for the new session)

Four serious bugs surfaced and consumed compute. **Do not repeat any of these.**

1. **Iteration 1 (~9h wasted on CNN auto+cross compressor)**: code bug — main() accessed `split_overlap_info["shared_example_count"]` which is TFDS-specific; harmonic audit returns `overlap_count` instead. Crashed AFTER compressor finished. Fix: line 4063 now uses `.get(..., .get(...))` fallback. Compressor checkpoints were preserved on disk and the NDE was recovered.

2. **Iteration 2 L1 launcher missing 5 SNR flags** (`--l1-min-snr -13`, `--l1-max-snr 13`, `--l1-min-snr-cross -5`, `--l1-max-snr-cross 5`, `--cross-snr-percentile 1.0`). Flag-diff tool caught the absent fields but interpreted them as "default-fallback" not "missing" — partially fixed by passing them, then we discovered `--l1-min-snr-cross/max-snr-cross` aren't even valid CLI flags (they're function parameters).

3. **PCA on by default in L1 script** (`--pca-components 50` default). PCA applied to L1 datavectors before NDE craters FoM3 by 5×. Fixed: launcher now passes `--pca-components 0`. **Documented as hard rule in `feedback_never_pca_l1.md`.**

4. **L1 cross with TFDS+--cross-maps silently uses broken `auto_scalar` noise model**. The fix `--cross-noise-model channel_empirical_global` is ONLY implemented for the harmonic-cache route. TFDS route prints a one-line warning and continues with auto_scalar. FoM3 craters 4× (40k → 10k). Fixed: launcher now uses `--full-sphere-cross-cache` for L1 cross. **Documented in `feedback_l1_cross_must_use_harmonic_route.md`.**

**Common pattern**: silent fallbacks to broken behavior, surfaced only by careful per-meta diff against a known-good anchor. The enhanced flag_diff tool now has explicit gotcha checks for #3 and #4.

---

## When the L1 cross runs land — checklist

The new session should do these in order, NOT skip any:

1. **Verify each L1 cross run's stdout contains both signals of correctness**:
   ```bash
   for s in 41 42 43; do
       log=scripts/sbi/results/exploratory/canonical_anchors_refresh/logs/l1_cross_canon_s${s}.log
       grep "cross_noise_model = channel_empirical_global" "$log" || echo "FAIL s${s}: harmonic noise model NOT applied"
       grep "CALIBRATING CHANNEL NOISE σ" "$log" || echo "FAIL s${s}: channel noise calibration not run"
   done
   ```
   If either fails, the run is broken and the rest of the steps are pointless.

2. **Verify each meta.json**:
   - `pca_applied: false`
   - `cross_maps_route: harmonic` (or similar — look for the harmonic route confirmation)
   - `npe_epochs: 50000`
   - `nde_train_split: train`

3. **Compute the headline numbers**:
   ```
   conda run -n jaxili python -c "..."  # see existing analysis scripts for template
   ```
   Expected ballpark for L1 cross: per-seed 30-45k, pool 30-35k, haircut 0.85-0.95 (matching v2_chsigma's 33.8k pool).

4. **If numbers look right**, re-run the comprehensive plotter:
   ```
   conda run -n jaxili python scripts/sbi/results/exploratory/canonical_anchors_refresh/tools/plot_canonical_diagnostics.py \
       scripts/sbi/results/exploratory/canonical_anchors_refresh
   ```
   This produces per-arm + per-seed detail pages. Send the PDF/PNGs to Andreas.

5. **If numbers look wrong** (e.g. L1 cross pool < 15k), do not relaunch. Investigate: check meta, check log, run flag-diff against v2_chsigma. Surface findings before any new compute.

---

## Tools the new session should use

| tool | what it does | when |
|:---|:---|:---|
| `flag_diff.py` | per-meta launcher diff with PCA + harmonic-route gotcha checks | **before any new compute** |
| `plot_canonical_diagnostics.py` | comprehensive per-arm + per-seed diagnostic plots | after analysis |
| `felt show <id>` / `felt history <id>` | inspect fiber state | session start |
| `felt edit <id> --outcome ...` | update outcomes as evidence lands | after each completed run |

---

## DO NOT

- DO NOT relaunch any of the 9 clean posteriors (CNN auto, CNN cross, L1 auto). They're correct.
- DO NOT kill the running L1 cross jobs unless an obvious crash signal appears in the logs.
- DO NOT launch arm 5 (sanity check) without checking with Andreas first — it's lower priority.
- DO NOT skip the flag_diff pre-flight check before any new compute.
- DO NOT silently retry — surface findings to Andreas before relaunching.
- DO NOT commit posteriors / logs / training dirs (CLAUDE.md §working-tree discipline).

---

## Open felt fibers

- `canonical-anchors-refresh-2026-05` (status: open) — THIS campaign
- `canonical-anchors-refresh-2026-05/code-extend-harmonic-slicing` (status: open) — code change shipped; close after final results
- `canonical-anchors-refresh-2026-05/sanity-check-auto-channel-tfds-vs-cache` (status: open) — arm 5 not run yet
- `cnn-h2-data-limit-scoping-2026-05` (status: open) — scoping only, no compute pending
- `cnn-h1-inductive-bias-2026-05` (status: closed)
- `cnn-h3-summary-dim-cdim100-test-2026-05` (status: closed)
- `cnn-auto-push-18-20-2026` (status: closed)

---

## Files I edited this session that are uncommitted

- `scripts/sbi/npe_cnn_nbody_tomo.py` — harmonic split slicing + audit; `auto_only` channel mode; KeyError fix at line 4063
- `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py` — added `--nde-train-split` flag
- `scripts/sbi/results/exploratory/canonical_anchors_refresh/launch_canonical_all.sh` — all four canonical arms with PCA off + correct routes
- `scripts/sbi/results/exploratory/canonical_anchors_refresh/tools/flag_diff.py` — enhanced with gotcha checks
- `scripts/sbi/results/exploratory/canonical_anchors_refresh/tools/plot_canonical_diagnostics.py` — comprehensive per-arm + per-seed plotting
- `.felt/canonical-anchors-refresh-2026-05/` — fiber + sub-fibers

The new session should commit these once the L1 cross results land cleanly. No commits yet.
