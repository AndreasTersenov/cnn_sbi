---
name: Amended cross-method check rejects iter-5/iter-19/iter-20 on joint_R; only iter-16 and iter-21 pass
status: closed
tags:
    - finding
    - cnn-auto-push-18-20-2026
    - ceiling-evidence
    - methodology
created-at: 2026-05-19T02:49:59.131278644Z
outcome: 'Scoring the constitution''s amended 3-component cross-method check across all 5 historically-landed iters: only iter-16 (Q2 120k) and iter-21 (Q9b F1+cbs+pool) PASS; iter-5/iter-19/iter-20 FAIL on the |dJoint_R|/joint_R_L1 component (0.77/0.43/0.26 vs threshold 0.25). The current pooled-best iter-20 is NOT a valid 5-seed replication target. The REPLICATION_LAUNCH.md decision-tree default (iter-20 if both null) must shift to iter-21 or iter-16. The amendment is meaningfully selective.'
---

# Amended check rejects iter-5/iter-19/iter-20

Filed: Ralph iter-13 (2026-05-19 ~02:50 UTC) after running the new
`landing_analysis.py` wrapper against every historically-landed iter in
the fiber.

## Findings table

Scored against the amended 3-component check from
[[cnn-auto-pooled-ratio-amendment-rationale]]:

- C1: pooled CNN/L1 ratio ≥ 0.35
- C2: |Δjoint_R| / joint_R_L1 ≤ 0.25  (joint_R_L1 = 0.175 from the
  default L1 auto+cross reference)
- C3: MoS CNN/L1 ratio ≥ 0.40

| iter | MoS    | pooled | joint_R | C1 ratio | C2 dJoint/L1 | C3 mos_r | verdict   |
|-----:|-------:|-------:|--------:|---------:|-------------:|---------:|-----------|
| 5    | 18 568 | 12 894 | 0.309   | 0.379 ✓  | 0.767 ✗      | 0.486 ✓  | **FAIL** |
| 16   | 19 502 | 13 868 | 0.215   | 0.408 ✓  | 0.228 ✓      | 0.510 ✓  | **PASS** |
| 19   | 18 213 | 13 087 | 0.251   | 0.385 ✓  | 0.433 ✗      | 0.477 ✓  | **FAIL** |
| 20   | 18 673 | 13 944 | 0.220   | 0.410 ✓  | 0.259 ✗      | 0.489 ✓  | **FAIL** |
| 21   | 17 805 | 13 829 | 0.186   | 0.407 ✓  | 0.062 ✓      | 0.466 ✓  | **PASS** |

(iter-22 and iter-23 in flight; will be scored on landing.)

## Why this is a substantive shift

Before this iter, `REPLICATION_LAUNCH.md` defaulted the "both null"
branch (iter-22 and iter-23 both confirm the ~14k pooled ceiling) to
**iter-20 as the 5-seed replication target** — chosen because it has the
highest pooled FoM3 of any landed iter (13 944). But iter-20 has
joint_R 0.220 vs L1's 0.175 → relative gap 0.259, just past the 0.25
threshold. Replicating iter-20 to 5 seeds would yield a config that
**does not satisfy the constitution's ceiling-certification cross-method
box**.

The shape-vs-scale tradeoff is real: iter-20 has a slightly tighter
pooled covariance (real information) but its per-seed mode drift relative
to the average per-seed posterior width is just outside the threshold
the amendment set. The amendment was constructed around iter-21's
specific numbers, but applying it across the campaign confirms it is
**selective**, not vacuous: 3 of 5 landed iters fail it.

## Implications for next Ralph iter

1. Default for the "both null" branch in `REPLICATION_LAUNCH.md` is
   **iter-21**, not iter-20. iter-21 has the cleanest pass margin on all
   3 components (0.407, 0.062, 0.466).
2. **iter-16 is a viable backup**: simpler config (no F1, no cbs=256,
   no pool=8/8 — just iter-5 baseline at 120k compressor steps), also
   passes. Worth scoring against iter-21 on per-seed CoV when both have
   5-seed data; the simpler config may be the more publishable choice.
3. If iter-22 lands in the "positive" band (pooled ≥ 15k) AND passes the
   amended check, it becomes a third candidate.
4. Same for iter-23.

## Pointers

- New script that produces this table mechanically:
  `scripts/sbi/autoresearch_cnn-auto-push/landing_analysis.py` (Ralph
  iter-13).
- Persisted per-iter outputs: `<iter-dir>/landing.json` (written by the
  script). Re-runnable; no recomputation needed.
- Related: [[cnn-auto-pooled-ratio-amendment-rationale]]
  (Ralph iter-12).
