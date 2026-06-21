# Goal-1 D1 — BNT deep-channel rescue in the MATCHED best-NDE pipeline (gated)

Pipeline: VMIM-MLP -> 10-D -> sbi_lens RealNVP 4x128, seeds 41,42,43, n=9000 median.
Gate C: TARP-DRP (600 val pts, dims=3) + SBC, pooled 3 seeds.
Registered: recovered=(deep2-BNT)/(noBNT-BNT); >=0.8 AND deep2 PASS gate => rescue confirmed.
MAF-ladder reference (old path, FoM3-only): noBNT 2405 / BNT 364 / deep2 2573 (rec 1.082).

| arm | FoM3 n=1000 | FoM3 n=9000 | gate verdict |
|---|---|---|---|
| nobnt_auto | 2375 | 2437 | - |
| bnt_auto | 424 | 425 | PASS-with-caveat |
| bnt_auto_deep2 | 3450 | 3498 | FAIL |

**recovered (n=9000) = (3498 - 425)/(2437 - 425) = 1.528**
