# CNN BNT/No-BNT ResNet Split Campaign Report

## Scope

- Branch: `l1-jax-resnet`
- Dataset: `NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48`
- Split policy: compressor train `train[:70%]`, NDE train `train[70%:]`, strict disjoint exact examples (`(cosmology, patch)`) enforced.
- Conditions: matched `no-BNT` vs `BNT` retraining/evaluation.
- Configs:
  - `control_plain_split` (plain CNN, 5 seeds)
  - `resnet_small_split` (handcrafted residual CNN, 5 seeds)
  - `resnet50_split` (Haiku ResNet50, 3 seeds)

## Execution note

During campaign execution, `resnet50_split` initially failed because compressor checkpoints were only saved at `save_every` intervals and `batch5000` was missing.  
Fixes applied:

1. `npe_cnn_nbody_tomo.py`: always save a final compressor checkpoint at the last completed step.
2. `run_cnn_bnt_losslessness_campaign.py`: robust checkpoint resolution for evaluation (uses latest compatible checkpoint if exact step file is unavailable) and records resolved checkpoint steps in metrics.

Final `resnet50_split` run used exact `batch5000` checkpoints for both conditions.

## Main quantitative result

The best ResNet campaign config by the campaign score is **not** a ResNet variant; it is `control_plain_split`.

| Config | Std inflation (BNT/no-BNT) | FoM ratio (BNT/no-BNT) | Rank score (lower better) |
|---|---:|---:|---:|
| control_plain_split | 1.0592 | 0.6099 | **0.4493** |
| resnet50_split | 1.0703 | 0.5791 | 0.4912 |
| resnet_small_split | 1.1222 | 0.4118 | 0.7104 |

Reference baseline from final-paper run (`bnt_comparison_tomo4`):

- Std inflation: **1.8049**
- FoM ratio: **0.0948**

So all split-campaign variants are much better than baseline, but ResNet does not beat the split plain-CNN control on BNT/no-BNT agreement.

## Seed-matched comparison (41/42/43)

To compare fairly with `resnet50_split` (3 seeds), using seeds `41,42,43`:

- `control_plain_split`: inflation `1.0580`, FoM ratio `0.6196`
- `resnet50_split`: inflation `1.0703`, FoM ratio `0.5791`
- `resnet_small_split`: inflation `1.1248`, FoM ratio `0.4181`

Additional key point: `resnet50_split` strongly lowers absolute no-BNT FoM relative to control (`~0.419x` of control on seeds 41–43), i.e. reduced overall information retention.

## Conclusion

Under the independent `(cosmology, patch)` split regime on multipatch data, the tested ResNet compressors **did not improve** BNT/no-BNT contour agreement versus the matched plain-CNN control, and `resnet50` notably reduced absolute constraining power.  

Current best configuration from this campaign remains:

- **`control_plain_split`** (plain CNN, split regime, matched retraining).

## Key artifacts

- Campaign summary: `campaign_summary.json`
- Per-config metrics: `control_plain_split/metrics.json`, `resnet_small_split/metrics.json`, `resnet50_split/metrics.json`
- Consolidated comparison: `comparison_resnet_summary.json`, `comparison_resnet_summary.csv`
- Overlays: `figures/overlay_baseline_finalpaper_combined_bnt_vs_nobnt.png` and per-config overlays under each config `figures/`
