# CNN-optimization results — L1 vs CNN on flat-local 10°/80px (2026-06)

Single durable reference for the CNN-side optimization that **reverses M1**. All numbers are
pooled 9000-obs median FoM3 (= 1/√det C₃ over [Ωm, σ8, w0]) unless noted, read from disk.

Branch: `cnn-nde-optimization-2026-06`. Working dir:
`scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/`.

---

## Headline

With a **per-summary best calibrated NDE**, the CNN-VMIM summary is **modestly better than L1+product**,
and the difference is small enough to read as "L1 is near-sufficient." Under BNT the CNN is **lossless**
while the L1 norm **collapses**. This flips the pre-optimization "L1 beats CNN" reading of M1.

The win is **not** hostage to the density estimator: each summary gets its own best calibrated flow.
L1's best NDE is the MAF (the same RealNVP that lifts the CNN *craters* on the 2000-D L1 vector), and
there is no evidence further NDE tuning helps L1.

> Not the old 10° "definitive CNN≥L1" result — that one was full-sphere cross-map **leakage** and is
> unrelated to this. This result is on de-leaked flat-local data, auto-only.

---

## The three findings

**1. It was the density estimator, not the CNN.**
On *frozen* plain-CNN summaries, swapping the readout flow from the common jaxili MAF to a sbi_lens
RealNVP jumped FoM3 **2312 → 3139** (+36%), calibrated and seed-robust. The same RealNVP craters on the
2000-D L1 vector (→1249), so the gain is CNN-specific, not a generic flow upgrade.

**2. A better compressor adds a bit more.**
A resnet18 compressor (over plain CNN) → **3326** (seed 41); 3-seed mean **3304**. ~15% over L1+product
(2875). Calibrated and seed-robust (3326 / 3314 / 3273).

**3. The best pipeline is BNT-lossless.**
resnet18+RealNVP under BNT: 3186 / 3164 / 3240 (mean **3197**), ratio **0.97×** — essentially lossless —
while L1 collapses to 0.15× (auto) / 0.22× (+product). In BNT space the CNN beats L1 by ~5–9×.

---

## All numbers, one place

### Main FoM3 (pooled 9000-obs median)

| Summary | NDE | no-BNT | BNT | BNT ratio |
|---|---|---|---|---|
| L1 auto-only | MAF | 2405 | 364 | 0.15× |
| L1 +product | MAF | **2875** | 637 | 0.22× |
| L1 +both (product+conv) | MAF | 2910 | — | — |
| CNN auto-only (resnet18+RealNVP) | RealNVP | 3326 / 3314 / 3273 (mean **3304**) | 3186 / 3164 / 3240 (mean **3197**) | **0.97×** |

CNN seeds are compressor seeds 41 / 42 / 43. Headline single-seed = 3326 (seed 41).

### NDE-family sweep (frozen plain-CNN summaries, seed 41)

| NDE family | FoM3 |
|---|---|
| jaxili MAF (old common readout) | 2312 |
| jaxili RealNVP | 2258 |
| jaxili MDN | 2885 |
| **sbi_lens RealNVP** | **3139** |

### Architecture sweep (compressor arch + sbi_lens RealNVP readout, seed 41)

| Compressor arch | FoM3 |
|---|---|
| plain | 3139 |
| **resnet18** | **3326** |
| plain + attention | 3205 |
| resnet_small | 3072 |
| resnet50_gn | 2760 |

### Apples-to-apples: L1 through each NDE family (the control)

| Summary | MAF | sbi_lens RealNVP | MDN |
|---|---|---|---|
| L1 (best config) | **2875 / 2861** | 1249 (craters) | 2549 |

The lift is CNN-specific; the same RealNVP that helps the 10-D CNN summary destroys the 2000-D L1 vector.

### Calibration (GATE C: TARP-DRP net-bias; + conservative, − over-confident)

| Arm | net-bias | HIGH-tercile |
|---|---|---|
| CNN auto-only (resnet18+RealNVP), no-BNT | +0.029 | +0.004 |
| L1 +product (MAF) | +0.000 | — |
| CNN BNT | — | +0.005 |

SBC: flat within the 99% binomial band for all three params (see fig 08). Both summaries calibrated.

---

## Final figures

Flat copies live in `paper_figures/` (pdf + png). Sources:

| # (paper_figures/) | source | what it shows |
|---|---|---|
| 01_corner_l1_vs_cnn_fiducial | `nde_sweep_2026_06_13/figs/corner_resnet18_fiducial` | L1 vs CNN corner at noise-avg fiducial |
| 02_corner_l1_vs_cnn_rep_patch | `…/corner_resnet18_rep_patch` | L1 vs CNN at a representative patch |
| 03_corner_l1_vs_cnn_stacked | `…/corner_resnet18_stacked` | population-stacked corner |
| 04_fom3_distribution_l1_vs_cnn | `…/fom3_distribution_resnet18` | per-obs FoM3 distributions |
| 05_calibration_best_cnn | `…/calibration_best_cnn_resnet18` | best-CNN calibration panel |
| 06_tarp_cnn_vs_l1_calibrated | `…/tarp_cnn_vs_l1_calibrated` | both summaries on the TARP diagonal |
| 07_tarp_cnn_standalone | `gate_c/tarp_drp/figures/tarp_resnet18_rnvp_dim3` | CNN TARP (un-stratified) |
| 08_sbc_cnn_rank_histograms | `gate_c/sbc/sbc_rank_histograms_resnet18` | CNN SBC, 99% band |
| 09_corner_cnn_bnt_vs_nobnt | `…/corner_cnn_bnt_vs_nobnt` | CNN BNT vs no-BNT (losslessness) |
| 10_fom3_bars_bnt_collapse | `…/fom3_bars_l1_cnn_bnt` | FoM3 bars: L1 collapses, CNN lossless |

`…/` = `nde_sweep_2026_06_13/figs/`.

---

## Key scripts (branch `cnn-nde-optimization-2026-06`)

- `scripts/sbi/train_nde_from_compressed.py` — NDE-family sweep on frozen summaries (family-preserving NaN retry).
- `scripts/sbi/tarp_stratified_val_nde.py` — GATE C for arbitrary NDE families.
- `scripts/sbi/run_nde_sweep.py`, `run_compressor_arch_overnight.py`, `run_compressor_arch_stage2.py` — orchestrators.
- `scripts/sbi/build_fiducial_summaries_cnn.py` — now `--compressor-arch` aware; supports `--flatsky-bnt`.
- `scripts/sbi/run_resnet18_bnt.sh` — BNT test for the best pipeline (= no-BNT recipe + `--flatsky-bnt`).
- Figure scripts: `corner_*`, `makefig_*`, `makefigs_best_cnn_resnet18.py`, `update_cnn_calibration_figs.py`.

## Caveats

- CNN seeds are compressor seeds; quote the 3-seed mean (3304 / 3197), not seed-41 alone, for the bars.
- FoM3 is fragile (1–2% correlation → ~50% swing); report σ/2D areas alongside. Calibration is the gate.
- Different NDE families per summary is intentional and fine (RealNVP for low-D CNN, MAF for high-D L1),
  given both pass GATE C.
