# A1 expressivity sweep — baseline anchor

Frozen resnet18 s41 summaries; reported posterior = sbi_lens RealNVP, 3 NDE seeds pooled.

| config | FoM3 (9000-obs) | un-strat TARP net (proper 1σ) | SBC (Ωm/σ8/w0) | reading |
|---|---|---|---|---|
| **4×128 (current)** | 3326 (s41) / 3304 (3-seed mean) | +0.033 ± 0.020 | 0.290/0.289/0.282 | mildly conservative |
| joint ℓ1 (target) | 3371 | +0.004 | 0.299/0.298/0.298 | joint-cal, marginals over-conf |

Goal: an expressive RealNVP config with TARP net ∈ [−0.02,+0.02], SBC ∈ [0.282,0.296], FoM3 ≥ 3371
(clean win) or ∈ [3304,3371) (tie). Guard: reject net < −0.02 or SBC > 0.305. Sweep on s41 first.
