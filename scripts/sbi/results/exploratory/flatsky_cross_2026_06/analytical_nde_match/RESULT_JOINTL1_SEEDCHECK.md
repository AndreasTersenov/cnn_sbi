# jointl1_nobnt — 3-seed COMPRESSOR robustness (Q1 winner)

VMIM(seed s) → sbi_lens RealNVP 4x128 (NDE 41,42,43 pooled, n=9000) → GATE C.
Reference: ℓ1+product 3045, CNN 3326.

| compressor seed | FoM3 n=9000 | gate |
|---|---|---|
| 41 | 3754 | PASS-with-caveat |
| 42 | 3761 | PASS-with-caveat |
| 43 | 4034 | PASS-with-caveat |

**3-seed band: 3754–4034 (mean 3850, spread 7%)**
