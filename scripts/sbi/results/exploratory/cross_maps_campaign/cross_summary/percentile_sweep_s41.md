# Cross-SNR percentile sweep — s41 only

Single-seed probe of `--cross-snr-percentile` to bracket the pct=1.0 setting
that drove the headline campaign. All runs share the same configuration
(multipatch 20deg/160px, `--zero-mean-maps`, 4 auto + 6 cross channels),
so the only varying knob is the percentile used to calibrate the cross
channels' SNR-bin range.

## BNT regime

Baseline `auto_zm_bnt` s41 FoM3 = 722.76.

| pct | s41 FoM3 | Δ vs auto_zm |
|---|---:|---:|
| min/max  | 429.75 | −41% |
| 5.0      | 633.96 | −12% |
| 1.0      | **875.67** | **+21%** |
| 0.5      | 754.26 | + 4% |
| 0.1      | 711.18 | − 2% |

Sweet spot: **pct=1.0**. Tighter percentiles cut into the bulk of the cross
distribution, and looser ones spend bins on the heavy outlier tail.

## no-BNT regime

Baseline `auto_zm_nobnt` s41 FoM3 = 12804.82.

| pct | s41 FoM3 | Δ vs auto_zm |
|---|---:|---:|
| min/max  | 2444.26  | −81% |
| 1.0      | 10932.42 | −15% |
| 0.5      | 1812.39  | −86% |
| 0.1      | 2136.41  | −83% |

No percentile setting recovers the auto-only no-BNT baseline. pct=1.0 is the
best of a bad set; pct=0.5 and pct=0.1 collapse FoM3 back to the original
min/max regime. Combined with the still-substantial 23–52% per-channel
zero-fraction at pct=1.0 (visible in `datavectors_nobnt_pct1.png`), the
inescapable read is that the no-BNT cross-map L1 channels carry no
extractable signal beyond what the auto channels already encode.

## Conclusion

Cross-map L1 channels add information **only in the BNT regime**, where BNT
removes most of the auto-channel cross-information and leaves the explicit
cross channels to fill the gap. In the no-BNT regime the auto channels'
SNR distribution at each scale already encodes the cross-bin correlations
implicitly, and adding cross-map L1 channels is a net distractor for the
flow regardless of percentile choice.
