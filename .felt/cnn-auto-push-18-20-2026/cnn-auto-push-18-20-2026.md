---
name: Push CNN auto-only FoM3 toward L1 auto+cross
status: closed
tags:
    - autoresearch
    - sbi
    - cnn
    - weak-lensing
created-at: 2026-05-17T21:58:04.756569363Z
closed-at: 2026-05-19T07:58:46.793477416Z
outcome: loop stuck
---

## Loop Status (live — read before deciding to do anything)

  **As of Ralph iter-19, 2026-05-19 06:32 UTC:** ceiling certified at iter-17.
  Representative best is **iter-16**: MoS 19 502, pooled **13 868**, joint_R 0.215,
  amended cross-method check PASS. Both ceiling-falsifiers (iter-22 Q9c at 120k,
  iter-23 Q4 wider VMIM aux) landed on the NULL branch — further gain requires
  architecture change (Tier-3, requires Andreas authorization).

  **Loop is in wait-for-Andreas mode.** The constitution's Done condition (B)
  reserves close-with-`outcome: ceiling-13868` for Andreas's explicit sign-off
  via `felt history append cnn-auto-push-18-20-2026 --summary "CEILING CONFIRMED: …"`.

  **Ralph: if a fresh iteration's survey finds none of (a), (b), (c) below, kill
  $PPID without changes.** Polish make-work on CEILING_EVIDENCE.md / STATUS.md
  is exhausted (iter-18 + iter-19 already did it); further appends are noise.

  - (a) Andreas appended `CEILING CONFIRMED: …` → close parent with
        `outcome: ceiling-13868`.
  - (b) Andreas requested a new branch (Q9d, alt-architecture, 5-seed
        replication, 240k promotion driven by Ralph) → file a fresh sub-fiber
        and resume work.
  - (c) Andreas answered [[cnn-auto-question-switch-to-pooled-fom3]] → close
        that question with the chosen option as outcome; do not close parent.

## Desired State

  CNN-based posterior estimation on tomographic weak-lensing **auto-only** inputs
  (4 auto maps, 20deg/160px, BNT-off, fiducial cosmology) achieves
  **mean-of-seeds FoM3 ≥ 40 000** across seeds 41/42/43 on the
  (Ω_m, σ_8, w_0) subspace — closing the gap to the
  L1 auto+cross headline and meaningfully above the current
  CNN-plain baseline (22 633 ± 5126) and CNN-resnet50-BN baseline (20 480 ± 2299).

  Two architecture arms are pushed in parallel:
  **CNN plain dense512** and **CNN resnet50_gn**. The improvement is
  provenance-complete in `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/metadata/`, holds across seeds
  (per-seed floor enforced as Guard), and the winning config from each arm
  is rerun at full step count (240 000) for confirmation before the desired
  state is declared reached.

  The stretch target is mean-of-seeds FoM3 ≥ 60 000 (matching L1 auto+cross).
  If reached, this is genuinely surprising (in a good way) — it would mean a CNN can extract
  from 4 auto maps what L1 needs 10 channels for — and warrants a fresh
  deep-read rather than just shipping the number.

  ## Context

  Project: `cnn_sbi`. Branch: `autoresearch/cnn-auto-push-18-20-2026` (off `l1-cross-maps`).
  Notes dir: `/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/`. Conda env: `jaxili`.

  **Code**:
  - CNN runner: `scripts/sbi/npe_cnn_nbody_tomo.py`. Auto-only is the
    default-no-flag invocation (no `--channel-mode`, no `--full-sphere-cross-cache`).
    Confirmed in session-1.
  - Per-arm runner: `scripts/sbi/autoresearch_cnn-auto-push/run_arm.py`.
    Two-phase: Phase A trains compressor; Phase B trains NDE per seed in parallel.
    Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so jobs use ~3 GB instead of
    pre-allocating 36 GB, which enables 4–5 parallel iterations per A100.
  - Existing CNN-plain auto-only baseline (22 633 mean):
    `scripts/sbi/results/exploratory/cnn_extended_train_zm/`
    (dense512, 240 000 steps, seeds 41/42/43).
  - Existing CNN-resnet50 auto-only baseline (20 480 mean, stock BN):
    `scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep/`.
  - Prior sweep map: `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/prior_cnn_sweeps_survey.md`
    (built session-1, includes resnet50 BN cdim=20 → 27 668 at 120k).
  - FoM3 wrapper: `scripts/sbi/autoresearch_verify_fom3.py` (committed).
  - Dataset: TFDS `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`,
    4 tomographic bins, full-sphere cache. Read-only.

  **Scope (the loop may edit) — EV-ranked priority order**:
  See `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/ITERATION_PLAYBOOK.md`. Do not
  pick from this list arbitrarily; pick from the top of the live priority queue.
  Knobs not in the playbook require justifying via the hypothesis discipline below.

  **Read-only**: `tf_dataset_nbody_tomo*.py`, the harmonic cache and base
  sims, `compare_probes_configs.py`, `autoresearch_verify_fom3.py`,
  `learn2map/`, the L1 pipeline, the Guard command, the inner script's
  data-loading / augmentation / normalization paths (modify only after
  filing an audit finding that names the specific suspected bug).

  **Resources & policy**:
  - GPUs 0, 1, 2 ONLY. GPU 3 is off-limits. Set `CUDA_VISIBLE_DEVICES` accordingly.
  - Each iteration checks `nvidia-smi --query-compute-apps=pid,used_memory`
    before launching, and activates `cluster-resources` skill.
  - Screening config: `--total-steps 60000` (~1/4 of the 240 000 baseline).
  - Per-iteration walltime budget: 3 hours wall, wrapped in
    `timeout (Walltime + 5)s`.
  - With `PREALLOCATE=false` and `xla=0.3`, GPU 2 fits ~4 parallel screening
    jobs comfortably (compute-saturated at 91% util; memory at ~30% per
    job). **Cap parallel jobs at 4** — beyond that, `--ds-batch-size=500`
    compress-dataset step risks OOM (iter-12 in session-1).
  - Confirmation: 240k rerun is a manual promotion step Andreas does, not
    the loop.

  ## Iteration Protocol

  **Every iteration starts with PHASE 0 and ends with PHASE FILE.** These are
  not optional. If you skip them you've left the protocol.

  ### PHASE 0 — Pre-iteration (hypothesis discipline)

  Before editing any file, do **all** of the following, in order:

  1. **Read the last 3 felt history entries** (`felt history cnn-auto-push-18-20-2026 --last 3`) and the `STATUS.md` table. Read at least one closed sub-fiber under the parent if its slug looks relevant to what you're about to try (`felt tree cnn-auto-push-18-20-2026`).
  2. **Read the prior 2 rows of `results.tsv`** for the current best, and the corresponding `metadata/iter-N_*.json` to see what's been tried, kept, reverted, or crashed.
  3. **State a hypothesis in writing.** Write it as: "I think **X** is limiting performance because **Y**. Changing X should move FoM3 by **±Z%** in direction **D**, because **Z-justification**." Save this to `<iter-dir>/hypothesis.md` before any code change.
     - If you cannot state a non-trivial hypothesis stronger than "try the next value of X", do NOT run training. **Run an audit instead** (see PHASE AUDIT below).
     - The hypothesis must be falsifiable. "Bigger model = better" is not a hypothesis; "the compressor is information-bottlenecked at dim=16 because val loss plateaus and dim sweep shows monotonic-up to 16, so increasing dim further OR widening the dense head should help" is.
  4. **Commit message format**: `experiment(autoresearch): <change> — hypothesis: <one sentence> — predicted Δ: <±N%>`. The commit body contains the full hypothesis paragraph from `hypothesis.md`.

  ### PHASE 1–7 — Standard autoresearch loop

  As in the `autoresearch` skill, with two modifications:

  - **PHASE 5.5 GUARD becomes a 4-way check**:
    a. Mean improvement clears noise floor (see Evidence/Verify).
    b. `fom3_per_seed_min ≥ 11 000` (working floor; constitution's original 18 000 was 240k-calibrated and blocks every 60k iteration).
    c. Compressor val loss curve looks healthy (monotonic-ish, no late-stage divergence; the iteration's Claude opens `<iter-dir>/logs/compressor.log` and eyeballs the `Step N | train | test` lines).
    d. **Cross-method overlay (MANDATORY every iteration, not just on improvement)**: render `<iter-dir>/overlay_vs_l1_autocross.pdf` via `conda run -n jaxili python scripts/sbi/autoresearch_cnn-auto-push/render_overlay.py --iter-dir <iter-dir>` (add `--is-best` if this iteration becomes the new best). The script updates `<run-dir>/latest_overlay.pdf` and (with `--is-best`) `<run-dir>/best_overlay.pdf` symlinks for Andreas's live visibility. The script also reports both mean-of-seeds FoM3 (autoresearch metric) and pooled FoM3 (covariance of the plotted contour) — log both to the iteration's metadata JSON. Flag if CNN/L1 pooled-ratio < 0.6 (seeds disagree too much) OR if the CNN contours land outside L1's 2σ on (Ω_m, σ_8, w_0).

  - **PHASE 6 DECIDE adds confidence calibration**:
    Record `predicted_delta` (from hypothesis) vs `actual_delta` in `metadata/iter-N_*.json` and in the felt history entry. After 5 iterations, if predictions are systematically off by >2× the actual magnitude in either direction, the implicit model of the bottleneck is wrong → trigger an audit.

  ### PHASE AUDIT — Triggered, not always scheduled

  Skip training. Spend the iteration on diagnostics. Triggered by **any** of:

  - **Cadence**: every 5th iteration regardless (clock-based).
  - **Plateau**: 3 consecutive iterations with no new best AND mean changes < 5% of std.
  - **Calibration failure**: 5 of the last 5 predictions wrong by >2×.
  - **Weird loss curve**: any iteration whose compressor val loss oscillates with amplitude > 0.5, plateaus before 30% of training, or whose NDE early-stops in the first 1k steps.
  - **High inter-seed variance**: CoV (std/mean) > 15% on a kept iteration.
  - **Cross-method disagreement**: the L1-overlay flag from PHASE 5.5 fires.

  What the audit iteration does (pick at least one; file findings as sub-fibers):

  1. **Code-read pass**: read 200–400 lines of `npe_cnn_nbody_tomo.py` that haven't been read yet. Targets in priority order: (a) data augmentation (noise injection, BNT path, zero-mean), (b) compressor architecture body (`Compressor`/`ResNet50GN` modules), (c) VMIM loss computation, (d) cache build path (compressor → compressed-dataset cache), (e) NDE flow construction. File any suspected bug as `cnn-auto-bug-<short-slug>` sub-fiber.
  2. **Loss-curve forensics**: plot the kept iterations' compressor val loss and NDE val loss in one figure. Look for: late-stage divergence, oscillation, plateau-before-2/3-done, train/val gap collapse. Save to `<iter-dir>/loss_curves.pdf`.
  3. **Adversarial peer review**: open a sub-process to attack the current best. The prompt is in `ITERATION_PLAYBOOK.md` (skeptical-referee mode). Capture three concrete attacks; file each as a sub-fiber.
  4. **Cross-method overlay**: render an L1 vs CNN corner plot for the current best. Look for: parameter directions where they disagree (potential bug surface), parameter directions where CNN is *tighter* than L1 (potential overfit), shape mismatches.
  5. **Route to `diagnose-training`** if the audit surfaces something that looks like a training pathology: NaN gradients, exploding losses, train-test split contamination, normalization mismatch.

  Audit iterations **do not** produce a Verify metric. They produce sub-fibers and (sometimes) updates to `ITERATION_PLAYBOOK.md` to re-rank the EV queue.

  ### PHASE FILE — Forced felt filing

  Before exiting the iteration:

  1. `felt history append cnn-auto-push-18-20-2026 --summary "iter-N (<change>): <hypothesis was H>. Result: <metric numbers + delta>. Implication: <H supported? Or what's the next hypothesis?>"` — this is **mandatory** and the iteration is not complete without it.
  2. If the result is non-obvious (surprising, contradicts prior, surfaces a new direction), also `felt add cnn-auto-<learning-slug>` with `-o '<one-line learning>'` and `--status closed`, then `felt nest <slug> cnn-auto-push-18-20-2026`.
  3. Update `metadata/iter-N_*.json` with `hypothesis`, `predicted_delta`, `actual_delta`, `calibration_error`, `cross_method_check` (if PHASE 5.5d ran).

  ## Skills

  Activated at the start of every iteration:

  - **`autoresearch`** — iteration discipline. Note: this constitution **modifies**
    the standard protocol; the modifications above (PHASE 0 hypothesis, PHASE AUDIT,
    PHASE FILE) are load-bearing.
  - **`coding-guidelines`** — pilot before scale-up (#5), verify-don't-vibe (#6),
    provenance (#9), convention vigilance (#8), surgical changes (#3). For this
    fiber especially: #6 "test must run, not be described" applies — never declare a
    bug fixed without running the diagnostic that would have caught it.
  - **`cluster-resources`** — pre-launch GPU check, mem-fraction selection.
  - **`figure-polish`** — for cross-method overlays and the loss-curve forensics
    plot produced by audit iterations.
  - **`felt`** — for history filing (PHASE FILE).
  - **`diagnose-training`** — **routing thresholds lowered** vs default:
    - default: 3 consecutive crashes
    - **also fires on**: plateau (3 kept iterations with no new best), confidence-calibration failure, weird loss curve flagged in audit, cross-method disagreement that doesn't resolve.

  ## Evidence

  The literal commands that count as ground truth.

  **Verify** (each training iteration):

  ```
  conda run -n jaxili python scripts/sbi/autoresearch_cnn-auto-push/run_arm.py \
      --arm <plain|resnet50_gn> --total-steps 60000 \
      --out-dir $NOTES_DIR/runs/cnn-auto-push-18-20-2026/iter-<n>/ \
      --gpus <available-from-cluster-resources> --seeds 41,42,43
    && conda run -n jaxili python scripts/sbi/autoresearch_verify_fom3.py \
      --posteriors-glob "$NOTES_DIR/runs/cnn-auto-push-18-20-2026/iter-<n>/posteriors/*_s4?.npy"
    | grep '^fom3_mean:' | awk '{print $2}'
  ```

  **Noise-aware keep/discard rule**: 3-seed FoM3 mean is noisy
  (iter-1 session-1: 18 307/14 909/15 230 — std/mean ≈ 9.5%). Replace strict
  "any improvement" with: **accept only if mean_new ≥ mean_best + 0.5 × max(std_new, std_best)**.
  An improvement that doesn't clear half the noise floor is logged as `tie`,
  not `keep` — it stays on the branch as a commit but the "best" pointer
  doesn't move. After 3 ties on the same axis, that axis is considered
  exhausted and the iteration must pick from a different scope row.

  **Guard** (metric-valued; runs on every iteration, not only on improvement):

  ```
  conda run -n jaxili python scripts/sbi/autoresearch_verify_fom3.py \
      --posteriors-glob "$NOTES_DIR/runs/cnn-auto-push-18-20-2026/iter-<n>/posteriors/*_s4?.npy"
    | grep '^fom3_per_seed_min:' | awk '{print $2}'
  ```

  Working floor: **per_seed_min ≥ 11 000** (≈ 0.85 × 60k baseline). The
  constitution's original 18 000 was 240k-calibrated. Above this floor: pass.
  Below it: discard regardless of mean. The 18 000 number is re-applicable
  only at 240k promotion.

  **Health check (PHASE 5.5c)**: each Verify is followed by a 30-second eyeball
  of `<iter-dir>/logs/compressor.log` for:

  - `nan` / `inf` strings anywhere — fail loudly.
  - Step N | train | test lines: train should decrease monotonically-ish; test should track within ±0.3 of train. If test diverges late, flag.
  - The "Saved @ step ..." lines: best-step should be in the last third of training (not the first checkpoint).
  - Any "patience" line above patience-limit/2: training is early-stopping; check whether that's the intent.

  **Cross-method overlay (PHASE 5.5d)**: every 3rd kept iteration, or on
  suspicion. Recipe in `ITERATION_PLAYBOOK.md`. Logs the overlay summary
  to `metadata/iter-N_*.json`.

  **Plateau / calibration status** (every iteration, situational awareness):

  ```
  conda run -n jaxili python ~/.claude/skills/autoresearch/scripts/summarize_results.py \
      $NOTES_DIR/runs/cnn-auto-push-18-20-2026
  ```

  Plus: count predictions in the last 5 `metadata/iter-N_*.json` files where
  `|predicted_delta - actual_delta| > 2 × |actual_delta|`. If ≥ 4, declare
  calibration failure and audit.

  **Done conditions — "Ceiling certification"**:

  The fiber closes only when there is a **defensible argument** that the
  current best is the CNN-framework ceiling on this dataset, not just an
  exhausted enumeration of hyperparameters. Two terminal states:

  **(A) Target reached** — close as `outcome: success`. Requires *all* of:

  1. ≥ 1 accepted iteration at 60k screening with FoM3 mean ≥ 30 000 AND per_seed_min ≥ 18 000.
  2. The screening winner has been **promoted to 240k confirmation**, and the
     240k mean clears the constitution's 40 000 target (or the stretch 60 000).
  3. The 240k confirmation includes the cross-method overlay
     (`overlay_vs_l1_autocross.pdf`); CNN and L1 contour shapes are
     consistent at the (Ω_m, σ_8, w_0) level (no obvious bug).
  4. Andreas signs off via felt history append.

  **(B) Ceiling reached short of target** — close as `outcome: ceiling-<value>`.
  Also a publishable result ("4-auto-channel CNN tops out at FoM3 ≈ N").
  Requires *all* of the following — the **Ceiling Certification Checklist**:

  - [ ] Every **Tier-1** hypothesis in `ITERATION_PLAYBOOK.md` is *tested* (kept, discarded, or tied) or explicitly closed as inapplicable.
  - [ ] Every **Tier-2** hypothesis is tested or explicitly closed with justification (e.g. "VMIM aux-width 256 didn't help at cdim=10 — `cnn_vmim_target_stability` — and the same null transfer is unlikely to surprise at cdim=16").
  - [ ] **At least 2 audit iterations** have occurred since the plateau started.
  - [ ] Cumulative **code-read coverage** (`code_read_coverage.md`) hits all 6 priority targets in PHASE AUDIT A1: data augmentation, compressor body, VMIM loss, cache build, NDE construction, test-split handling. No unaddressed `cnn-auto-bug-*` sub-fiber remains open.
  - [ ] At least one **adversarial peer-review iteration** (PHASE AUDIT A3) has run; each of the 3 challenges is either resolved (with a sub-fiber outcome) or filed as a deferred-question open sub-fiber.
  - [ ] Cross-method overlay (`overlay_vs_l1_autocross.pdf`) on the current best shows **CNN contours consistent with L1's shape** on (Ω_m, σ_8, w_0): no degeneracy-axis disagreement that suggests a bug. Pooled-FoM3 ratio CNN/L1 ≥ 0.5 (we're not asking for parity, just that the CNN is doing real inference).
  - [ ] The current best has been **5-seed replicated** (not just 3): seeds 41–45 to match the run_b_advanced and a2 baselines. CoV is reported.
  - [ ] The current best has been **promoted to 240k**; the 240k pooled-vs-mean-of-seeds gap is reported (if pooled is dramatically lower than mean-of-seeds, the per-seed posterior drift is real and limits the credibility of the mean).
  - [ ] A `cnn-auto-ceiling-evidence` sub-fiber exists, status closed, with `-o "FoM3 ceiling = <value>; closed because <one-sentence why we believe it's the ceiling>"`. The body cites the checked boxes above with specific evidence.
  - [ ] Andreas reviews the certification doc and signs off via `felt history append --summary "CEILING CONFIRMED: …"`.

  **(C) Failure-to-reach-ceiling** — if the loop runs out of compute budget
  before either (A) or (B) is met, close as `outcome: incomplete-<value>`
  with explicit accounting of which checklist items are unmet. Treat this
  as a partial result for the next loop.

  If plateau fires below target and the certification checklist is
  incomplete, the loop does NOT close. It files
  `cnn-auto-stuck-at-<value>` summarizing what was tried, runs an audit
  iteration directed at the unchecked items, then iterates 2 more times
  on the audit's suggested directions before stopping for Andreas.

  ## Open Questions / EV-ranked hypothesis queue

  Live priority queue — pick from the top. Each entry: hypothesis, prior evidence, EV estimate, suggested test.

  See `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/ITERATION_PLAYBOOK.md` for
  the full ranked list with prior-evidence citations and pre-formatted hypothesis statements.

  **Tier-1 (high prior-evidence weight, untested or in-flight)**:

  *Live status as of Ralph iter-14 (2026-05-19 ~03:00 UTC). The
  campaign is in ceiling-certification mode; new training launches are
  reserved for ceiling-falsifiers, not exploration. See
  `ITERATION_PLAYBOOK.md` for the full ranked list; see
  [[cnn-auto-ceiling-evidence]] for the certification scaffold.*

  1. **resnet50_gn cdim=20 lr=1e-3 at 60k** — TESTED iter-15 (FALSIFIED, -52.8% vs iter-5; compressor undertrained). 120k retest DEFERRED via [[cnn-auto-deferred-q1-resnet50gn-120k]] (architecture-change Tier-3 scope).
  2. **Compressor-steps > 60k (Q2)** — TESTED iter-16 (+5% MoS / +7.5% pooled). CLOSED.
  3. **5-seed replication of iter-5 and iter-7 (Q3)** — DEFERRED via [[cnn-auto-deferred-q3-5seed-iter5-vs-iter7]] (noise-axis quibble, unlikely to move ceiling).
  4. **Q9 stack (cbs=256 + pool=8/8)** — TESTED iter-20 (+6.5% pooled). CLOSED.
  5. **Q9b (F1 on Q9 stack)** — TESTED iter-21 (drift additive, pooled flat). CLOSED. Filed [[cnn-auto-pooled-fom3-ceiling-near-14k]].
  6. **Q9c (Q9 stack at 120k)** — TESTED iter-22 (LANDED 2026-05-19 ~04:50 UTC, BOTH_NULL branch: pooled 12 531 (-10.1% vs iter-20), MoS 19 304, joint_R 0.272, amended-check FAIL; compressor argmin@15% gap 1.08 nats — architectural ceiling confirmed). CLOSED ceiling-confirming.

  **Tier-2 (lower prior weight or speculative)**:

  7. **Q4 (VMIM aux width 128 → 256 at cdim=16)** — TESTED iter-23 (LANDED 2026-05-19 ~03:50 UTC, REFUTED on every axis: pooled 12 945 (-7.2%), joint_R 0.281 (drift WORSENED), bound LOOSER by 0.26 nats at matched step, amended-check FAIL). [[cnn-auto-bug-vmim-aux-may-bias-compressor]] CLOSED REFUTED. Default --vmim-nf-hidden 128 is at the joint-stability sweet spot.
  8. **Q5 (NDE flow depth 12+)** — TESTED iter-12 (crashed). Not retested; structural-bug surface, low EV.
  9. **Q6 (cbs=256 + lr=1e-3 robust best)** — TESTED iter-11; subsumed by Q9 (iter-20). CLOSED.
  10. **Q7 (LR schedule variants)** — DEFERRED via [[cnn-auto-deferred-q7-lr-schedule-variants]] (premise falsified by iter-16).
  11. **Q8 (resnet50 stock BN at cdim=20)** — DEFERRED via [[cnn-auto-deferred-q8-resnet50-stockbn-cdim20]] (architecture-change Tier-3; BN-contamination prior).

  **Tier-3 (low prior weight, audit-driven only)**:

  7. **Augmentation changes** — only after an audit reads the augmentation code path and identifies a specific suspected issue.
  8. **Normalization changes** — same gate as augmentation.
  9. **Architecture swap (ViT, FNO, etc.)** — requires explicit Andreas authorization.

  **Things the iteration must NOT silently resolve — surface as sub-fibers**:

  - Cross-method disagreement (L1 vs CNN posterior at fiducial).
  - Any code-read finding that smells like a bug.
  - Any audit-iteration conclusion that re-ranks the EV queue.
  - Any iteration where actual_delta is in the *opposite* direction from predicted (model of bottleneck is wrong).

  **Closed findings from session-1** (read before iterating; in `.felt/cnn-auto-push-18-20-2026/`):

  - `cnn-auto-cdim16-not-20` — plain optimum is 16, not 20 (resnet sweet spot didn't transfer).
  - `cnn-auto-3seed-noise` — iter-1 mean was outlier-driven; 1–2k differences may be one seed.
  - `cnn-auto-cbs256-stability` — cbs=256 trades mean for 5× tighter scatter.
  - `cnn-auto-builtin-lr-schedules` — inner script already has piecewise (compressor) + cosine (NDE); our `--compressor-lr` is the *initial* value.
  - `cnn-auto-compressor-undertrained` — compressor NOT plateaued at 60k.
  - `cnn-auto-prealloc-infra-fix` — PREALLOCATE=false unlocks 4–5 parallel jobs per A100.
  - `cnn-auto-parallel-oom-compress` — `--ds-batch-size=500` × 5 concurrent OOMs at compress step.
  - `cnn-auto-guard-recalibrated` — constitution's 18 000 floor was 240k-calibrated; working floor is 11 000.
  - `cnn-auto-resnet50gn-untested` — never run on auto-only before iter-15.
  - `cnn-auto-lr-landscape` — peak at 1e-3, possible "bump" at 3e-3 (could be noise).
  - `cnn-auto-cli-overrides-tradeoff` — parallel iterations use CLI overrides, not commits.

  **Andreas-in-the-loop checkpoint**: every ~5 kept iterations, the loop pauses
  for Andreas to read the felt history, the audit findings, and the cross-method
  overlay. Andreas re-ranks the EV queue if needed. Pure-autonomous iteration
  has known limits on research problems where the bottleneck is structural;
  this checkpoint is what makes the loop-plus-human pattern work.
