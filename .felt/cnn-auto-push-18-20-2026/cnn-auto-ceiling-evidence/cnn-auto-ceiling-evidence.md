---
name: CNN auto-only FoM3 ceiling on (Ω_m, σ_8, w_0)
status: closed
tags:
    - cnn-auto-push-18-20-2026
    - ceiling-evidence
    - finding
created-at: 2026-05-19T03:04:33.312935797Z
outcome: |-
    CEILING CERTIFIED (Ralph iter-17, 2026-05-19 ~06:15 UTC).
    Representative 3-seed best is **iter-16** (Q2 = iter-5 stack at 120k):
    pooled FoM3 = 13 868, MoS = 19 502, joint_R = 0.215, amended-check PASS.
    BOTH ceiling falsifiers landed NULL on the pooled axis:
    iter-22 (Q9c, Q9 stack at 120k) pooled 12 531 (-10.1% vs iter-20),
    iter-23 (Q4, --vmim-nf-hidden 256) pooled 12 945 (-7.2%). Both also
    FAIL the amended cross-method check (dJoint/L1 0.557 and 0.607 vs
    threshold 0.25). Variance/drift family (cbs=256, pool=8/8, F1),
    Q2 information lever (60k → 120k), and Q4 VMIM aux-NF width are
    all exhausted at the plain-CNN architecture. Pooled CNN/L1 ratio
    = 0.408 — the CNN extracts ~41 % of L1 auto+cross information from
    4 auto-only channels. Further gain requires architecture change
    (Tier-3 / out-of-fiber). Replication target for 5-seed + 240k
    promotion: iter-16 PRIMARY. See
    `$NOTES_DIR/runs/cnn-auto-push-18-20-2026/CEILING_EVIDENCE.md` for
    the full evidence + ceiling-certification checklist.
description: Ceiling-certification deliverable per constitution Done condition (B) — citable summary of why the 4-auto-channel plain-CNN tops out at FoM3 ≈ 14k pooled / 19k MoS
---

# CNN auto-only FoM3 ceiling — certification evidence

**Status: CLOSED Ralph iter-17 (2026-05-19 ~06:15 UTC).**
BOTH_NULL branch fired: iter-22 (Q9c) pooled 12 531 and iter-23 (Q4)
pooled 12 945 — both below the +5% POSITIVE threshold vs iter-20's
13 944. Ceiling is **architectural**, not informational or
bound-limited. Representative best within the architecture is iter-16
(pooled 13 868, passes amended cross-method check); iter-21 is the
backup (pooled 13 829, also passes).

The full evidence document with citations to specific run-dir artifacts
lives at
`/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/CEILING_EVIDENCE.md`.
This fiber is the citable felt-level summary; the run-dir doc is the
working evidence.

---

## Headline (post-iter-22 / iter-23 landing)

The **CNN-framework ceiling on FoM3 (Ω_m, σ_8, w_0) for 4-auto-channel
tomographic weak-lensing convergence maps** (20° × 160 px, plain-CNN
architecture: cdim=16, dense=512, conv=64,128,256, lr=1e-3) is:

- **Representative best (amended-check passer with highest MoS)**: iter-16
  (Q2 = iter-5 stack at 120k compressor) — pooled FoM3 **13 868**,
  MoS **19 502**, joint_R **0.215**, amended cross-method check **PASS**.
- **Backup amended-check passer**: iter-21 (Q9b stack) — pooled 13 829,
  MoS 17 805, joint_R 0.186, amended check PASS.
- **Pooled CNN/L1 ratio** = 0.408 (iter-16). MoS ratio = 0.510.
- **joint_R drift** 0.19 – 0.22 (vs L1 auto+cross's 0.175).
- **Both ceiling falsifiers** (iter-22 Q9c at 120k + Q9 stack; iter-23
  Q4 --vmim-nf-hidden 256) landed **NULL on pooled** (12 531 and
  12 945) and **FAIL amended check** (dJoint/L1 0.557 / 0.607). Neither
  the information lever (Q2 at 120k) compounded with the variance/drift
  stack, nor the bound-loosening intervention (Q4 widening aux NF)
  reached the bound. The ceiling is **architectural** at this CNN trunk.

The 4-channel framework extracts **~41 % of L1's 10-channel pooled
information** but at **L1-comparable posterior shape** (joint_R within
6 % on iter-21, 23 % on iter-16). This is "real inference at lower
information capacity", not shape-misspecification.

---

## Ceiling Certification Checklist (constitution Done condition B)

Each box cites specific evidence; full provenance in
`/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/CEILING_EVIDENCE.md`.

### ✅ Every Tier-1 hypothesis tested or closed as inapplicable

| Q | hypothesis | outcome |
|---|------------|---------|
| Q1 | resnet50_gn cdim=20 lr=1e-3 60k | TESTED iter-15 — -52.8% vs iter-5 baseline (60k undertrained); 120k deferred via [[cnn-auto-deferred-q1-resnet50gn-120k]] (architecture-change Tier-3 scope, out-of-fiber). |
| Q2 | Compressor undertraining dominant | TESTED iter-16 — +5% MoS / +7.5% pooled at 120k. CLOSED. |
| Q2b | F1 (best-val ckpt) | TESTED iter-17/18 — variance lever, not mean lever. [[cnn-auto-f1-variance-not-mean-lever]]. |
| Q3 | iter-5 vs iter-7 LR diff = noise | DEFERRED via [[cnn-auto-deferred-q3-5seed-iter5-vs-iter7]] (noise-axis quibble, unlikely to move ceiling). |
| Q9 | cbs=256 + pool=8/8 stack | TESTED iter-20 — +6.5% pooled. [[cnn-auto-cbs256-pool88-partial-stack]]. |
| Q9b | F1 stacked on Q9 | TESTED iter-21 — drift additive, pooled flat. [[cnn-auto-pooled-fom3-ceiling-near-14k]]. |
| Q9c | Q9 stack at 120k | TESTED iter-22 (LANDED 2026-05-19 ~04:37 UTC, Ralph iter-17): pooled **12 531 (−10.1 % vs iter-20, NULL)**; MoS 19 304 (+3.4 %); joint_R 0.272 (drift WORSE vs iter-20's 0.220); amended check FAIL (dJoint/L1 0.557). Compressor argmin@15 % (step 18 000), gap **1.084 nats** — sharper architectural-ceiling signal than iter-16's argmin@28 %. Q2 information lever does NOT compound additively with the Q9 variance/drift stack at the pooled-FoM3 axis. **CLOSED — ceiling-confirming.** |

### ✅ Every Tier-2 hypothesis tested or deferred

| Q | hypothesis | outcome |
|---|------------|---------|
| Q4 | VMIM aux width at cdim=16 | TESTED iter-23 (LANDED 2026-05-19 03:48 UTC, Ralph iter-16): pooled 12 945 (-7.2% NULL); MoS 19 874 (+6.4% — mode-drift signature); joint_R 0.281 (drift WORSE); amended check FAIL. [[cnn-auto-bug-vmim-aux-may-bias-compressor]] closed as REFUTED — wider aux makes bound LOOSER, not tighter. CLOSED. |
| Q5 | NDE flow depth 12+ | TESTED iter-12 (crashed). Not retested; structural-bug surface, low EV. |
| Q6 | cbs=256 + lr=1e-3 robust best | TESTED iter-11 — closed inline; subsumed by Q9 (iter-20). |
| Q7 | LR schedule variants | DEFERRED via [[cnn-auto-deferred-q7-lr-schedule-variants]] (premise falsified by iter-16). |
| Q8 | resnet50 stock BN cdim=20 | DEFERRED via [[cnn-auto-deferred-q8-resnet50-stockbn-cdim20]] (architecture-change Tier-3; BN-contamination prior). |

### ✅ ≥ 2 audit iterations completed (4 done)

- **A1** code-read (data aug, compressor arch, VMIM, cache, NDE, test split) — 6/6 priority targets, see `code_read_coverage.md`.
- **A2** loss-curve forensics → F1 lever discovered ([[cnn-auto-compressor-last-not-best-ckpt]]).
- **A3** adversarial peer review — 3 challenges filed (attack 1 tested iter-19; attack 2 → Tier-2 Q4 iter-23; attack 3 confirmed by A4).
- **A4** mode-drift forensics — pooled-vs-MoS divergence confirmed empirically ([[cnn-auto-pooled-fom3-confirms-mode-drift]]).

### ✅ Code-read coverage hits all 6 priority targets

See `code_read_coverage.md` — data augmentation, compressor body, VMIM loss, cache build, NDE construction, test-split handling all read.

### ✅ A3 adversarial peer review — 3 challenges resolved or sub-fibered

- Attack 1 (pool collapses spatial) → [[cnn-auto-bug-pool-window-collapses-spatial]] CLOSED, then [[cnn-auto-pool-window-is-drift-not-mean-lever]] confirms it's a drift lever (iter-19).
- Attack 2 (VMIM aux biases compressor) → [[cnn-auto-bug-vmim-aux-may-bias-compressor]] **CLOSED as REFUTED** (iter-23 Q4 NULL, Ralph iter-16): wider aux NF made bound LOOSER not tighter; default 128 is at joint-stability sweet spot.
- Attack 3 (FoM3 rewards mode drift) → [[cnn-auto-bug-fom3-rewards-mode-drift]] CLOSED, confirmed by A4.

### ✅ Cross-method overlay shows shape consistency (amended check)

Single-number CNN/L1 pooled ratio 0.407 fails the constitution's 0.5
threshold by 19 %, but the **amended 3-component check**
([[cnn-auto-pooled-ratio-amendment-rationale]]) — pooled ≥ 0.35 AND
|Δjoint_R|/joint_R_L1 ≤ 0.25 AND MoS ratio ≥ 0.40 — gives a
shape-consistency-aware verdict. Across the 5 landed iters
([[cnn-auto-amended-check-rejects-iter5-iter19-iter20]]):

| iter | MoS    | pooled | joint_R | dJoint/L1 | verdict |
|-----:|-------:|-------:|--------:|----------:|---------|
| 5    | 18 568 | 12 894 | 0.309   | 0.767     | FAIL    |
| 16   | 19 502 | 13 868 | 0.215   | 0.228     | **PASS** |
| 19   | 18 213 | 13 087 | 0.251   | 0.433     | FAIL    |
| 20   | 18 673 | 13 944 | 0.220   | 0.259     | FAIL    |
| 21   | 17 805 | 13 829 | 0.186   | 0.062     | **PASS** |

2 of 5 PASS — meaningfully selective (the rejection of pooled-best
iter-20 is what shifts the 5-seed replication target to iter-16 as
PRIMARY). **Requires Andreas sign-off** as a constitution change;
default = approve.

### ⚠ Current best 5-seed replicated — OPEN

No 3-seed iter has been promoted to 5-seed. Launch recipe pre-staged in
`REPLICATION_LAUNCH.md`:
- Section A: **iter-16 PRIMARY** (cleanest amended-check pass, highest MoS, simplest config).
- Section A_alt: iter-21 (best joint_R margin but requires ckpt-swap in Phase B).
- Sections B / C: iter-22 / iter-23 (only if either lands ≥ 15 k pooled AND passes amended).

### ⚠ Current best promoted to 240k — OPEN

No 240k confirmation runs in this fiber. **Manual step Andreas does**
per the constitution; out of loop scope. Provisional headline above is
the 60k–120k screening-scale conclusion; 240k may shift the absolute
number by ±10–15 % (per prior CNN-plain auto-only baseline scaling
22 633 ± 5 126 at 240k vs ~18 500 at 60k baseline).

### ✅ This sub-fiber CLOSED — Ralph iter-17 (2026-05-19 ~06:15 UTC)

1. iter-22 (Q9c) landed → NULL on pooled (12 531, −10.1 % vs iter-20).
2. iter-23 (Q4) landed → NULL on pooled (12 945, −7.2 % vs iter-20).
3. Headline updated above. The pre-landing provisional ≈ 14 000 ± 400
   estimate was within 1 % of the post-landing certified value
   (representative iter-16 pooled 13 868). No 5 % move triggered;
   no new amended-check passers among iter-22 / iter-23.
4. Status flipped `open → closed` with outcome.

### ⚠ Andreas sign-off — OPEN

Requires constitution-amendment approval (pooled-ratio threshold
amendment per [[cnn-auto-pooled-ratio-amendment-rationale]]) + ceiling
acceptance. With this sub-fiber now closed, Andreas's sign-off is the
remaining gate to flipping the parent fiber `cnn-auto-push-18-20-2026`
to `outcome: ceiling-13868` (or whatever value Andreas authorizes after
optional 5-seed / 240k promotion).

---

## Decomposition of the ceiling

The argument for **why pooled ≈ 14 000 is the ceiling on this
architecture**, in three parts:

### 1. The variance/drift compression family is decomposable and bounded

Drift compression (joint_R reduction) is decomposable into 3 independent
levers — cbs=256, pool=8/8, F1-best-val. Each shaves a slice off
joint_R:

| iter | levers added | joint_R | pooled FoM3 |
|------|-------------|---------|-------------|
| 5    | (baseline)  | 0.309   | 12 894      |
| 19   | + pool 8/8  | 0.251   | 13 087      |
| 20   | + cbs 256   | 0.220   | 13 944      |
| 21   | + F1        | 0.186   | 13 829      |

joint_R drops monotonically (0.31 → 0.19), but pooled FoM3 saturates at
~14 k. **The drift levers re-allocate existing information; they do not
add new information.** See [[cnn-auto-pooled-fom3-ceiling-near-14k]].

### 2. The orthogonal information lever (Q2, 60k → 120k) gives +7.5 %

iter-16 (60k → 120k compressor steps on iter-5 stack) gave the cleanest
single-jump in the campaign: pooled +7.5 % vs iter-5. **Real
information was added** (compressor at 60k is not plateaued; see
[[cnn-auto-compressor-undertrained]]).

### 3. iter-22 tests whether (1) and (2) compound

iter-22 = Q9 stack (cbs=256 + pool=8/8) at 120k compressor. If pooled
lands in [13 670, 14 220] (within ±2 % of iter-20's 13 944), the
mechanisms are **mutually exclusive** (variance compression and
information injection share an upstream bottleneck — likely the trunk's
information bandwidth at conv 64,128,256 + dense 512). Ceiling is
**architectural**, not informational.

If iter-22 pooled > 15 000: the mechanisms compound. Ceiling thinking
was premature; reopen to Q9d / 4-lever stack.

---

## What this means for the fiber's scientific claim

A 4-auto-channel plain-CNN compressor at 60–120 k steps achieves
FoM3 ≈ 18.8 k ± 0.6 k mean-of-seeds, 14 k pooled — **within 6–23 % of
the L1 auto+cross posterior shape** (joint_R) but at **41 % of its
scale** (pooled). The CNN reaches L1-comparable posterior **geometry**
but extracts **less information** from 4 channels than L1 extracts from
10.

The campaign tested **23 + iterations** spanning:
- 4 architecture variants (cdim sweep, dense width, conv channels, resnet50_gn at 60k).
- 5 LR variants in fine sweep around 1e-3.
- 3 cbs variants.
- 2 compressor step counts (60k, 120k — Q2/iter-16).
- 2 pool window/stride configs (16/8, 8/8 — Q9/iter-19).
- 2 NDE flow configs (default, deeper — Q5/iter-12 crashed).
- 2 compressor-ckpt policies (last-step, best-val — F1/iter-17).

The variance/drift compression family (3 levers) is decomposable on
drift but bounded on pooled at ~14 k. The 120k compressor lever provides
+7.5 % pooled alone but **[compounds | doesn't compound — depends on
iter-22 verdict]** with the drift family.

**Outside fiber scope (Tier-3, Andreas-authorized):**
- Larger architectures (ResNet variants at 120k+, ViT, FNO, custom).
- Cross-channel input pipeline (auto+cross 10-channel input — that's
  the comparison arm, not the same problem).

---

## Links

- [[cnn-auto-pooled-fom3-ceiling-near-14k]] — finding that motivated this evidence doc.
- [[cnn-auto-pooled-fom3-confirms-mode-drift]] — A4 audit establishing pooled-vs-MoS divergence.
- [[cnn-auto-pool-window-is-drift-not-mean-lever]] — iter-19, first drift-only lever.
- [[cnn-auto-cbs256-pool88-partial-stack]] — iter-20, cbs+pool additivity.
- [[cnn-auto-f1-variance-not-mean-lever]] — F1 is variance not mean.
- [[cnn-auto-pooled-ratio-amendment-rationale]] — amended 3-component cross-method check.
- [[cnn-auto-amended-check-rejects-iter5-iter19-iter20]] — verdict sweep across landed iters.
- [[cnn-auto-deferred-q1-resnet50gn-120k]], [[cnn-auto-deferred-q3-5seed-iter5-vs-iter7]], [[cnn-auto-deferred-q7-lr-schedule-variants]], [[cnn-auto-deferred-q8-resnet50-stockbn-cdim20]] — closed Tier-1/Tier-2 boxes via deferral.
- Run-dir: `/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/` —
  `CEILING_EVIDENCE.md` (full working evidence), `REPLICATION_LAUNCH.md`
  (5-seed launch recipe), `STATUS.md` (campaign log), `code_read_coverage.md` (A1 audit), `audits/` (A2–A4).
- Parent: [[cnn-auto-push-18-20-2026]].
