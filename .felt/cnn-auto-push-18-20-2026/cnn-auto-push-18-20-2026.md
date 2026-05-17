---
name: Push CNN auto-only FoM3 toward L1 auto+cross
status: open
tags:
    - autoresearch
    - sbi
    - cnn
    - weak-lensing
created-at: 2026-05-17T21:58:04.756569363Z
---
## Desired State

  CNN-based posterior estimation on tomographic weak-lensing **auto-only** inputs
  (4 auto maps, 20deg/160px, BNT-off, fiducial cosmology) achieves
  **mean-of-seeds FoM3 ≥ 50 000** across seeds 41/42/43 on the
  (Ω_m, σ_8, w_0) subspace — closing the gap to the
  L1 auto+cross headline and meaningfully above the current
  CNN-plain baseline (22 633 ± 5126) and CNN-resnet50-BN baseline (20 480 ± 2299).

  Two architecture arms are pushed in parallel:
  **CNN plain dense512** and **CNN resnet50_gn**. The improvement is
  provenance-complete in `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/metadata/`, holds across seeds
  (per-seed floor enforced as Guard), and the winning config from each arm
  is rerun at full step count (240 000) for confirmation before the desired
  state is declared reached.

  The stretch target is mean-of-seeds FoM3 ≥ 38 000 (matching L1 auto+cross).
  If reached, this is genuinely surprising — it would mean a CNN can extract
  from 4 auto maps what L1 needs 10 channels for — and warrants a fresh
  deep-read rather than just shipping the number.

  ## Context

  Project: `cnn_sbi`. Branch off `l1-cross-maps` as `autoresearch/cnn-auto-push-<date>`.

  **Code**:
  - CNN runner: `scripts/sbi/npe_cnn_nbody_tomo.py`. Auto-only is either
    the default-no-flag invocation or a `--channel-mode auto_only` flag —
    the first iteration must verify which and log it.
  - Existing CNN-plain auto-only baseline (22 633 mean):
    `scripts/sbi/results/exploratory/cnn_extended_train_zm/`
    (dense512, 240 000 steps, seeds 41/42/43). Read its meta + any
    CLI-args trail before proposing changes.
  - Existing CNN-resnet50 auto-only baseline (20 480 mean, stock BN):
    `scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep/`.
    BN was fine for auto-only per `memory/project_resnet_bn_contamination.md`
    (the contamination only bites on multi-channel harmonic input).
  - Prior sweeps: per Andreas, "for plain CNN a lot of things have been tried;
    the records should be there, but I don't know where exactly." The first
    iteration's REVIEW phase greps `scripts/sbi/results/` for `cnn_*` dirs,
    reads their `meta.json` / README / CLI logs, and files findings as
    sub-fibers (`felt add prior-cnn-sweeps-<arm> -s closed -o "<summary>"`).
    Subsequent iterations consult these before proposing changes.
  - FoM3 reference: `scripts/sbi/compare_probes_configs.py` (definition at
    the top). The autoresearch loop uses the wrapper
    `scripts/sbi/autoresearch_verify_fom3.py` (committed to the project)
    to extract the scalar without rerunning the full 3×3 table.
  - Dataset: TFDS `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`,
    4 tomographic bins, full-sphere cache. Read-only.
  - Conda env: `jaxili`. Every command runs through `conda run -n jaxili python ...`.

  **Scope (the loop may edit)**:
  - A per-arm runner script the loop creates at
    `scripts/sbi/autoresearch_cnn-auto-push/run_arm.py` (created in the pilot
    iteration; thin wrapper that parallelises 3-seed training on GPUs 0–2).
  - Hyperparameters: LR, batch size, total-steps (≤ 60 000 in screening mode),
    warmup, schedule, EMA, augmentation, compressor head dim/dense width,
    NDE flow depth/width, train/val arrangement.
  - Architecture variants WITHIN the chosen arm (resnet50_gn block widths,
    group counts; CNN plain layer widths, dropout, activations). NOT a swap
    to a different family (ViT, etc.) without Andreas's say-so.

  **Read-only**: `tf_dataset_nbody_tomo*.py`, the harmonic cache and base
  sims, `compare_probes_configs.py`, `autoresearch_verify_fom3.py`,
  `learn2map/`, the L1 pipeline, the Guard command.

  **Resources & policy**:
  - GPUs 0, 1, 2 ONLY. GPU 3 is off-limits (Andreas's policy). Set
    `CUDA_VISIBLE_DEVICES` accordingly.
  - Other users (notably `alahiry`) sometimes grab GPUs unannounced. Each
    iteration checks `nvidia-smi --query-compute-apps=pid,used_memory`
    before launching training, and (when available) activates
    `cluster-resources` for adapting GPU count + per-GPU memory fraction.
  - Screening config: `--total-steps 60000` (~1/4 of the 240 000 baseline).
    Per-iteration walltime budget: 3 hours. Autoresearch wraps Verify in
    `timeout (Walltime + 5)s` as a hard cutoff.
  - Confirmation: when the loop accepts a new best, it does NOT also retrain
    at full step count automatically. Full-step confirmation is a manual
    promotion step Andreas does at the end of the run.

  ## Skills

  `autoresearch` (the iteration discipline), `coding-guidelines`
  (verify-don't-vibe, pilot-before-scale-up, provenance), `cluster-resources`
  (GPU/CPU availability), `figure-polish` (for any corner-plot of a new
  best config vs. baseline). Activated at the start of every iteration.

  `felt` activates automatically inside Ralph; the loop files sub-fibers
  for findings worth keeping (prior-sweep discoveries, surprising failures,
  dead-end hypotheses).

  `diagnose-training` is on standby — autoresearch routes to it after
  3 consecutive crashes.

  ## Evidence

  The literal commands that count as ground truth.

  **Verify** (each iteration):
  conda run -n jaxili python scripts/sbi/autoresearch_cnn-auto-push/run_arm.py
      --arm <plain|resnet50_gn> --total-steps 60000
      --out-dir $NOTES_DIR/runs//iter-/
      --gpus 0,1,2 --seeds 41,42,43
    && conda run -n jaxili python scripts/sbi/autoresearch_verify_fom3.py
      --posteriors-glob "$NOTES_DIR/runs//iter-/posteriors/*_s4?.npy"
    | grep '^fom3_mean:' | awk '{print $2}'

  **Guard** (runs only when Verify improved): metric-valued, threshold
  18 000 (tunable in autoresearch's `config.md` if the baseline floor
  turns out higher or lower):
  conda run -n jaxili python scripts/sbi/autoresearch_verify_fom3.py
      --posteriors-glob "$NOTES_DIR/runs//iter-/posteriors/*_s4?.npy"
    | grep '^fom3_per_seed_min:' | awk '{print $2}'
  Loop accepts if value ≥ 18 000. Prevents "improved mean via wider-spread
  seeds" — the same trap that bit resnet50_gn auto+cross.

  **Soft signal** (logged, NOT gated): SBC rank-uniformity on a 100-sim
  hold-out using the existing SBC runner (commit `08575f6` in this branch).
  Recorded in provenance as `metric.guard_soft.sbc_ks_p` and appended to
  the iteration's `felt history append --summary` line. Per Andreas, SBC
  "is not very good as a metric and can be wrong" — do not discard on it,
  but a sharp drop warrants a human-readable flag in the summary.

  **Plateau status** (every iteration, situational awareness):
  conda run -n jaxili python ~/.claude/skills/autoresearch/scripts/summarize_results.py
      $NOTES_DIR/runs//

  **Done-conditions** (the fiber closes only when all hold): at least one
  accepted iteration has mean-of-seeds FoM3 ≥ 30 000 with per-seed floor
  ≥ 18 000; `summarize_results.py` reports a long plateau and the iteration
  judges further exploration unproductive (free-text justification logged
  to felt history); Andreas has been notified via fiber update so he can
  promote a winning config for full-step confirmation.

  If the plateau fires but the FoM3 target hasn't been met, the iteration
  files `cnn-auto-stuck-at-<value>` as a sub-fiber summarizing what's been
  tried and the open hypotheses, then continues — Andreas decides
  between-runs whether to raise the screening budget, change architectures,
  or stop.

  ## Open Questions

  Things the iteration must NOT silently resolve — surface as sub-fibers
  or in `felt history append --summary`.

  - Does `npe_cnn_nbody_tomo.py` support `--channel-mode auto_only`, or is
    auto-only the default-no-flag invocation? First iteration verifies +
    logs.
  - CNN-resnet50_gn on auto-only: worth running at all? The BN-contamination
    note says BN is fine on auto-only. If stock BN consistently beats GN on
    auto-only, that itself is a publishable methodological observation.
  - Screening-vs-full gap: does 60 000-step screening order configs
    correctly relative to 240 000? After 3–4 accepted configs, manually
    retrain one at full step count and compare ranking. If the gap is wild,
    raise the screening budget.
  - Is the 38 000 stretch target achievable, or is 4-auto-maps
    information-theoretically below 10-channel L1? A hard ceiling well
    below 38 000 is itself a scientific result.
  - Prior-sweep discovery: first-iteration finding lives here. If
    exhaustive prior sweeps are found, the loop's exploration directions
    are constrained accordingly.