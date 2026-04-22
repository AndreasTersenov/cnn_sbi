# CNN noiseless vs noisy comparison

- Baseline noisy root: `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/bnt_comparison_tomo4`
- Noiseless run root: `/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/final/paper_sbi_consolidation/cnn_noiseless_vs_noisy`
- Seeds: `41,42,43`

## Shrink/expansion summary

| condition | std_ratio (noiseless/noisy) | fom_ratio | sigma8_std_ratio | Om-s8 area_ratio |
| --- | ---: | ---: | ---: | ---: |
| `nobnt` | 0.6438 | 7.7962 | 0.4184 | 0.2265 |
| `bnt` | 0.3543 | 86.5925 | 0.0417 | 0.0223 |

Overlay figures:
- `figures/overlay_nobnt_noisy_vs_noiseless_combined.png`
- `figures/overlay_bnt_noisy_vs_noiseless_combined.png`