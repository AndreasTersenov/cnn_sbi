# Definitive L1 vs CNN — Phase C summary (perm 0, seeds pooled)

Primary metrics: **marginal σ** and **2D FoM** (FoM3 is fragile — [[feedback_fom3_fragile_use_2d_areas]]). CNN fast-route absolute treated as fine (overlap empirically negligible, Andreas 2026-05-31).

Arms with posteriors: 10

## Marginal σ (lower = tighter)

| arm | n | σ(Ωm) | σ(σ8) | σ(w0) | σ(h0) | σ(ns) | σ(Ωb) | FoM3 |
|---|---|---|---|---|---|---|---|---|
| L1 auto+cross | 3 | 0.0273 | 0.0423 | 0.1245 | 0.0420 | 0.0424 | 0.0054 | 34607 |
| L1 auto-only | 3 | 0.0393 | 0.0519 | 0.2039 | 0.0487 | 0.0496 | 0.0059 | 10560 |
| CNN-RealNVP auto+cross | 3 | 0.0268 | 0.0378 | 0.1508 | 0.0416 | 0.0385 | 0.0076 | 26748 |
| CNN-RealNVP auto-only | 3 | 0.0351 | 0.0420 | 0.2163 | 0.0512 | 0.0545 | 0.0078 | 9125 |
| CNN-MAF auto+cross | 3 | 0.0346 | 0.0416 | 0.2126 | 0.0469 | 0.0521 | 0.0081 | 11984 |
| CNN-MAF auto-only | 3 | 0.0433 | 0.0590 | 0.2165 | 0.0496 | 0.0546 | 0.0067 | 6679 |
| CNN-RealNVP auto+cross (std) | 3 | 0.0257 | 0.0385 | 0.1446 | 0.0404 | 0.0356 | 0.0064 | 24281 |
| CNN-auto native-TFDS (RealNVP) | 3 | 0.0298 | 0.0397 | 0.1484 | 0.0486 | 0.0499 | 0.0083 | 14969 |
| CNN auto+cross multi-perm (3 perms) | 9 | 0.0295 | 0.0559 | 0.1546 | 0.0427 | 0.0367 | 0.0077 | 7868 |
| CNN auto-only multi-perm (3 perms) | 9 | 0.0442 | 0.0578 | 0.2396 | 0.0526 | 0.0531 | 0.0078 | 6096 |

## 2D FoM (higher = tighter)

| arm | Ωm–σ8 | Ωm–w0 | σ8–w0 |
|---|---|---|---|
| L1 auto+cross | 2349 | 323 | 191 |
| L1 auto-only | 1133 | 169 | 102 |
| CNN-RealNVP auto+cross | 2190 | 358 | 197 |
| CNN-RealNVP auto-only | 1140 | 189 | 117 |
| CNN-MAF auto+cross | 1466 | 201 | 129 |
| CNN-MAF auto-only | 855 | 154 | 89 |
| CNN-RealNVP auto+cross (std) | 2187 | 372 | 200 |
| CNN-auto native-TFDS (RealNVP) | 1587 | 283 | 179 |
| CNN auto+cross multi-perm (3 perms) | 791 | 328 | 124 |
| CNN auto-only multi-perm (3 perms) | 829 | 150 | 87 |

## TARP coverage

See `tarp_2026_05_31/figures/tarp_{per_arm,overlay}_dim{3,6}.png` and `tarp_summary.json`. A calibrated arm sits on the diagonal; below = over-confident.

_Auto-generated; re-run aggregate_all_arms.py to refresh as arms land._
