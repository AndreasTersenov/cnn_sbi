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
outcome: 'OPEN. 10-arm comparison (3 L1 + 4 CNN-RealNVP-companion + 2 CNN-MAF-companion + 1 sanity check), jaxili MAF NDE for all, 3 seeds × 3 perms = 90 posteriors. Architecture: 64,128,256/dense=256/cdim=10, 80k steps, best-val. Primary metric: FoM3 (2D areas + marginal σ secondary). Plan: plans/mighty-tumbling-sparrow.md. Triggered by comprehensive-experiment-audit-2026-05.'
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

## Loop Status (live)

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
