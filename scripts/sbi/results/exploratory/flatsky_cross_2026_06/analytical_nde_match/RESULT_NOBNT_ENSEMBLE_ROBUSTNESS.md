# no-BNT analytical-arm ensemble robustness check (2026-06-27)

Question: does promoting the no-BNT per-channel l1 arms (auto, +product) to the 3-compressor
deep ensemble (uniform with joint l1 and the BNT arms) change M1? Driver: ensemble_eval.py
(--mode {product,auto}_nobnt), 9000-fiducial-obs pooled FoM3. CNN left single (already calibrated).

| arm     | no-BNT single | no-BNT ENSEMBLE | delta  | BNT ens | retention ens/ens |
|---------|---------------|-----------------|--------|---------|-------------------|
| auto    | 2448          | 2429            | -0.8%  | 388     | 0.16              |
| product | 3045          | 3009            | -1.2%  | 718     | 0.24              |
| joint   | 3371 (ens)    | 3371            |  0.0%  | 2424    | 0.72              |
| CNN     | 3326 (single) | 3326            |  --    | 3186    | 0.96              |

VERDICT: ensembling the no-BNT analytical arms changes FoM3 by <=1.2% (inside FoM3 fragility),
and the BNT retention is unchanged to 2 d.p. (0.16/0.24). So M1 is robust to single->ensemble,
and fig:bnt needs no change. The no-BNT singles are confirmed robust; the ensemble is the calibrated
estimator where it matters (BNT, where the single is over-confident). Caches: l1product_vmim_s4{1,2,3}
(pre-existing), l1none_vmim_s41 + ens_nobnt_auto_s4{2,3} (built here). Outputs:
{product,auto}_nobnt_ensemble/median_summary.json.
