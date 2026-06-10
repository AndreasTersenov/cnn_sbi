# GATE C — calibration of the de-leaked flat-local CNN posteriors (2026-06-10)

Companion to `FLATSKY_CNN_RESULT.md`. Establishes whether the de-leaked CNN result (CNN gains
nothing from the physical cross; L1 > CNN there) rests on *calibrated* posteriors, so the FoM3/σ
numbers can be trusted. Three complementary tests; the headline rests on the two load-bearing arms
(**auto-only** and **+product**), both of which pass all three.

The common jaxili MAF (the same NDE for CNN and L1) is what is being calibrated here — the CNN-VMIM
compressor feeds its 10-d summary into that MAF, so this validates the full CNN inference chain.

## The three tests (why all three)

| test | scope | what it catches | what it misses |
|---|---|---|---|
| **TARP-DRP** | global, joint (N-D) | mis-coverage of the joint credible regions, per FoM3 tercile | local effects that average out |
| **SBC** | global, marginal (per param) | rank non-uniformity = global over/under-confidence | a local bias at the fiducial that cancels over the prior |
| **L-C2ST** | **local, at the fiducial** | mis-calibration *specifically at the observed cosmology* | nothing local — this is the sharpest test for a real obs |

L-C2ST is the decisive one for a real measurement (you observe one cosmology, not the whole prior),
and — unlike the high-dim L1 case where its self-test was underpowered and it could not be run — it
**works for the CNN** because the summary is only 10-dimensional.

## Per-arm verdict

| arm | TARP (tightest tercile) | SBC | L-C2ST reject@p<0.05 | overall |
|---|---|---|---|---|
| **auto-only** | calibrated (\|dev\| 0.029) | mild-conservative | **0% (median p 0.27)** | ✅ calibrated |
| **+product** | calibrated (\|dev\| 0.037) | mild-conservative | **0% (median p 0.22)** | ✅ calibrated |
| +both | calibrated (\|dev\| 0.037) | mild | 13% (median p 0.12) | ◻ mostly ok |
| +conv | **over-confident (−0.068)** | mild | **60% (median p 0.05)** | ⚠ miscalibrated |

**Bottom line: the two arms the science rests on (auto-only, +product) are calibrated across all
three tests.** `+conv` is locally miscalibrated, but it is the throwaway arm — it gains nothing over
auto-only (FoM3 2192 vs 2325) and is the flat-sky analog of the alm-product that was ~99% leakage on
the L1 side; we do not use it for any claim. `+both` inherits a little of conv's issue (it contains
the conv channels) but stays mostly within tolerance.

## Reading the L-C2ST figure (`lc2st/lc2st_cnn.png`)

Each panel: the grey histogram is the **permutation-null** distribution of the L-C2ST statistic T
(what T looks like when the posterior *is* locally calibrated); the coloured ticks are the **30
observed** statistics T(x₀) at typical fiducial obs, with their median (solid line); the dashed line
is the p=0.05 reject threshold (95th percentile of the null).

- **auto-only, +product:** observed T(x₀) fall *inside* the null bulk, median well left of the
  threshold → 0% reject → locally calibrated.
- **+conv:** observed T(x₀) are shifted *right*, sitting on/past the threshold → 60% reject → the
  classifier can tell the posterior from the truth at the fiducial → locally miscalibrated.
- **+both:** mostly inside the null, a few past the line → 13%.

### Why the verdicts are trustworthy (the self-test power gate)

A "calibrated" L-C2ST is only meaningful if the test *could* have detected miscalibration. The
self-test does exactly this: it plants a known 0.5σ w₀ shift and checks the test rejects it.

- **ST_H0** (no planted error) median p = 0.71 → does not false-alarm.
- **ST_H1** (planted 0.5σ w₀ error) median p = 0.02 → **detected**.

So the test has power. `+product`'s 0% reject is a genuine pass, not a powerless one — which is the
exact failure mode that made L-C2ST inapplicable to high-dim L1
(`reference_lc2st_underpowered_highdim_l1`).

## The SBC nuance (conservative, not over-confident)

SBC ranks have mean ≈ 0.5 (no bias) but std 0.273–0.281, **below** the uniform 0.289 → the ranks
bunch toward the middle → the posteriors are mildly **conservative** (over-covering), consistent with
the TARP curves sitting on/above the diagonal. The KS test flags this as "non-uniform" on Ωm/σ8, but
it is the *safe* direction: conservative posteriors do not overstate constraining power, so if
anything the CNN's (already non-existent) cross gain is conservatively estimated — it cannot be an
artifact of over-confidence.

## Implication for the result

The de-leaked headline — **CNN extracts no information from the physical cross while L1 gains +20%,
so L1 > CNN on the cross; auto-only ties** — rests on the auto-only and +product posteriors, both
calibrated globally (TARP, SBC) and locally at the fiducial (L-C2ST, with demonstrated power). The
result is calibration-trustworthy.

## Artifacts
- TARP: `tarp_drp/figures/`, `tarp_drp/tarp_summary.json`
- SBC: `sbc/sbc_rank_histograms.{png,pdf}`, `sbc/sbc_summary.json`
- L-C2ST: `lc2st/lc2st_cnn.{png,pdf}` (built by `plot_lc2st_cnn.py`), per-arm `lc2st/*/lc2st_summary.json`
