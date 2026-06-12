# PLAN — GATE C (TARP + SBC) on the joint-statistic arms

**Date:** 2026-06-12. **Status:** awaiting Andreas's sign-off. **Author:** Fable 5 session.
**Campaign dir:** `scripts/sbi/results/exploratory/flatsky_cross_2026_06/overnight_menu/`.
**Rule being satisfied:** "never trust a contour before GATE C" — the joint-stat arms
(OVERNIGHT_RESULT.md addenda) have full-rigor FoM3/marginals and corner figures but NO
calibration. Nothing uncalibrated enters the paper.

---

## 1. Objective and decision metric

Decide, per joint-statistic arm, whether its posteriors are calibrated well enough to quote
in the paper (contours + marginals + the "≈ l1+product from autos alone" claim). Decision
metrics, both DERIVED from artifacts:

- **TARP-DRP** (stratified by per-point FoM3 terciles, 600 held-out val points, 3 MAF
  seeds): signed max |ECP − α| on the dim-3 science subspace, per tercile, worst seed.
- **SBC** (ranks pooled from the TARP dumps, n = 1800 per arm): per-science-param rank
  mean (uniform = 0.5) and std (uniform = 0.289); KS p reported but NOT gating (at n=1800
  it is hypersensitive — the BNT gate had p=0.000 on arms whose magnitude check passed).

L-C2ST is **deliberately skipped**: 3000-dim conditioning is exactly the regime where its
self-test power gate aborts (memory `reference_lc2st_underpowered_highdim_l1`); Andreas
already accepted TARP+SBC as GATE C for high-dim arms (flat-local campaign precedent).

## 2. Scope — which arms

| arm | dim | why | in scope |
|---|---|---|---|
| `pair2dq_nobnt` | 3000 | quotable contour: pair2d 2794 ≈ l1+product 2875, σ_s8 0.072 | YES (core) |
| `jointl1q_nobnt` | 3000 | quotable contour: jointl1 2788; "weighting adds nothing" claim | YES (core) |
| `pair2dq_bnt` | 3000 | BNT-basis corner figure exists (morning session); 0.52 invariance ratio | YES (recommended) |
| `jointl1q_bnt` | 3000 | same; 0.54 ratio | YES (recommended) |
| `full4dq_*`, `full4da_*` | 256/— | grid-transport MEASUREMENT arms (ratios 0.45/0.70); no contour of theirs is paper-bound | NO — ratios are relative statements between identically-trained twins; precedent: rescue-ladder arms (deep/deep2/whiten/unions) were not individually gated either |

Recommendation: run all four q-arms. Marginal cost of the two BNT-side arms is ~2 × 10 min
(measured basis below) and it covers BOTH bases of the corner figures already produced.

## 3. Machinery (adapt `run_bnt_gate_c.py`)

New one-purpose driver `scripts/sbi/run_joint_gate_c.py`, phases:

1. **tarp** — `tarp_stratified_val.py` per arm (greedy scheduler over GPU slots; corners
   phase from the BNT gate is DROPPED — the joint-arm corners already exist in
   `overnight_menu/corners/` + `figures/`, made by the morning session; lc2st phase
   DROPPED per §1).
2. **coverage** — `run_tarp_coverage.py --dims 3 6` on the dumps.
3. **sbc** — ranks from the TARP dumps (same `sbc_from_dumps` helper as the BNT gate).
4. **report** — `overnight_menu/gate_c/GATE_C_JOINT.md` with DERIVED verdicts only
   (same table format as `bnt_campaign/gate_c/GATE_C_BNT.md`, including the magnitude
   check translating worst deviation into % credible-interval misestimate).

Driver start-up asserts (fail-fast, before any GPU work):
- each arm's `cache/l1_cache_meta.npz` has `dequantize == True` and the expected
  `stat`/`basis`/`k=10`/`snr_range=5`;
- `l1_train.npz`/`l1_val.npz` exist with 6-param theta and 3000-dim x.

## 4. Configuration fingerprint (mirrors the full-rigor sweeps EXACTLY)

The GATE C NDE retrains must be the same model family + recipe that produced the quoted
numbers (`run_full4d_retry.py` → `population_sweep_flatsky.py`, full mode):

| knob | value | source |
|---|---|---|
| caches | `overnight_menu/<arm>/cache/l1_{train,val}.npz` | the quoted sweeps' inputs |
| preproc | `log1p-zscore`, clip 5, min-feature-variance 1e-5 | `run_full4d_retry.py:53` + sweep defaults |
| MAF seeds | 41,42,43 | `run_full4d_retry.py:54` (full mode) |
| NDE training | epochs 50000, batch 256, lr 1e-4, warmup 100, decay 10000 | shared defaults of `population_sweep_flatsky.py` and `tarp_stratified_val.py` |
| TARP points | n-points 600, m-samples 2000 | BNT-gate precedent (same statistical power) |
| env | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, mem-fraction per packing (§6) | BNT-gate precedent |

Dequantization lives IN the caches (meta `dequantize=True`), so it carries through
automatically — no re-dequantization at gate time.

## 5. Registered predictions (BEFORE data; bands fixed now)

- **P-G1 (noBNT joint arms):** behave like the gated l1 noBNT arms — all terciles
  |ECP−α| ≤ 0.05, SBC std within 0.289 ± 0.015, |mean − 0.5| ≤ 0.02. Band: PASS clean.
- **P-G2 (BNT-side joint arms):** mildly worse, like the BNT l1/cnn arms (BNT space is a
  harder learning problem — GATE_C_BNT precedent of mixed ±0.08): worst tercile
  |dev| ∈ (0.05, 0.10]. Band: PASS-with-caveat.
- **Sensitivity note (registered as part of the read):** the joint-stat headline
  ("marginals equal-or-better than l1+product", σ_s8 0.072 vs 0.075 = a 4% edge) is far
  more calibration-sensitive than pillar-2's 6.6× headline. The l1+product comparator
  passed GATE C at |dev| ≤ 0.037. Therefore the verdict for the "equal-or-better" claim is
  COMPARATIVE: if a joint noBNT arm shows systematic over-confidence (negative dev /
  SBC std > 0.289) at the ~4–5% level, the claim downgrades from "equal-or-better" to
  "comparable", regardless of the arm passing in isolation.

Verdict bands (per arm, derived):
- **PASS** — all terciles |dev| ≤ 0.05 AND SBC std ∈ [0.275, 0.305]: quotable, no caveat.
- **PASS-with-caveat** — worst tercile |dev| ∈ (0.05, 0.10] or SBC std outside by < 0.02:
  quotable with named caveat (tercile + direction), as was done for cnn-product-BNT.
- **FAIL** — |dev| > 0.10 or SBC std off by ≥ 0.02: contour does NOT enter the paper;
  investigate before any rerun.

## 6. Cost and GPU plan (measured anchor, not guessed)

Measured: in the BNT gate, each L1-type TARP arm (2000-dim, same row counts, same recipe)
took 470–520 s on one A100 (3 NDE retrains ≈ 80 s each + 600-point × 3-seed sampling).
These arms are 3000-dim → same order; call it ~8–15 min/arm. Four arms packed two-at-a-time
≈ 20–35 min wall + a few minutes for coverage + report. (The handoff's "~2 h" was a
conservative pre-estimate.)

GPU plan: at last check GPU 1 is free (4 MiB), GPUs 0/2 have foreign tenants (6.9 GB/31%
and 5.1 GB/65%), GPU 3 never. **Plan: pack 2 jobs on GPU 1** (`XLA_PYTHON_CLIENT_MEM_FRACTION
= 0.45` each, per the 0.9/N rule) rather than squeezing beside busy tenants. Re-run the
tenant check immediately before launch; if GPU 1 is taken, fall back to whichever of 0/2
has the lighter tenant with a 0.40 cap, surfacing the contention in the felt stanza.
CPU: well under the 50-worker budget (2 python processes).

Launch pattern (the one that works; absolute paths, no backgrounded cd-lists):
```
(cd /mnt/home/tersenov/software/cnn_sbi/scripts/sbi && setsid nohup \
  /home/tersenov/anaconda3/envs/jaxili/bin/python run_joint_gate_c.py --gpus 1,1 \
  > /mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/overnight_menu/gate_c/driver.out 2>&1 &)
```
then `pgrep -f "[r]un_joint_gate_c"` + absolute-path tail; re-arm a fresh Monitor
(driver.out diff + GATE_C_JOINT.md existence + pgrep liveness — the old session's monitors
are dead).

## 7. Post-run folds (all derived from artifacts)

1. `overnight_menu/gate_c/GATE_C_JOINT.md` — the report (written by the driver).
2. GATE-C status lines added to `OVERNIGHT_RESULT.md` (night-synthesis table) and the
   overnight stanza of `FLATSKY_BNT_RESULT.md`; adjudicate P-G1/P-G2 in this plan file.
3. Memory `project_joint_onepoint_stats_and_grid_transport` — append the gate verdict
   (it currently carries "GATE C pending on all overnight arms").
4. Felt fiber: one completion stanza (launch + verdict in one, given ~30 min runtime).
5. Commits by path: driver script, this plan, the report, coverage figures (png/pdf).
   NEVER the dumps/ckpts (`.npz`, orbax) — they stay on disk under `gate_c/`.

## 8. Decisions for sign-off

- **D1:** include the two BNT-side q-arms (recommended YES, §2)?
- **D2:** packing 2 jobs on GPU 1 vs 1 job each on GPUs 1 + (0 or 2) (recommended: GPU 1
  packed; zero contention risk with foreign tenants)?
- **D3 (only if you care):** n-points 600 / m-samples 2000 are the BNT-gate values;
  bumping them buys smoother curves at linear cost. Default: keep.

On "go": build `run_joint_gate_c.py`, dry-run print of commands, fresh tenant check,
detached launch, Monitor re-armed, folds on completion.
