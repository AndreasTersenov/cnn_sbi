# Table 1 — FoM3 with error bars (PAPER-READY)

Central values are the PUBLISHED ones. The +/- is the measured compressor-seed spread
(std over **three** independently trained compressors, pre-ensemble), expressed as an
absolute uncertainty on that central value. n=3 for EVERY row, so the column is
comparable across summaries.

| Summary | frame | FoM3 | (rel) | retrained check |
|---|---|--:|--:|--:|
| l1 auto | no-BNT | **2448 ± 27** | 1.1% | 2777 ± 31 |
| l1 auto | BNT | **388 ± 43** | 11.1% | 391 ± 43 |
| l1 +product | no-BNT | **3045 ± 183** | 6.0% | 3232 ± 194 |
| l1 +product | BNT | **718 ± 29** | 4.1% | 758 ± 31 |
| joint l1 | no-BNT | **3371 ± 96** | 2.8% | 3380 ± 96 |
| joint l1 | BNT | **2424 ± 208** | 8.6% | 2405 ± 207 |
| CNN | no-BNT | **3326 ± 14** | 0.4% | 3428 ± 14 |
| CNN | BNT | **3186 ± 19** | 0.6% | 3147 ± 18 |

Block-bootstrap SE of the median (subdominant; caption only): l1 auto, no-BNT ±5.0, l1 auto, BNT ±1.1, l1 +product, no-BNT ±6.1, l1 +product, BNT ±3.0, joint l1, no-BNT ±7.6, joint l1, BNT ±6.5, CNN, no-BNT ±8.4, CNN, BNT ±6.6.

± = spread over three independently trained compressors, per NOTE_FOM_ERROR_BARS.md §5.3-5.4;
the single→ensemble shift is the bias term, reported separately, never summed with the ±.

## Caveat measured with extra seeds (see SEEDCHECK_N6.json)

Six compressor seeds were run for the two BNT rows with the widest n=3 spread. The
three-seed value is retained above for cross-row comparability, but it is not the whole
picture for joint l1 BNT:

| row | n=3 | n=6 | n=6 range |
|---|--:|--:|---|
| l1 auto, BNT | ±49 (11.1%) | ±39 (8.8%) | 389–481 |
| joint l1, BNT | ±260 (8.6%) | ±416 (13.4%) | 2551–3712 |

l1 auto BNT tightened (11.1% -> 8.8%); joint l1 BNT WIDENED with every added seed
(8.6% -> 10.9% -> 13.4%, range 2551-3712, best/worst = 1.46x). For that row the n=3
bar understates the true compressor-to-compressor variability by roughly a third.
