# Multi-compressor-seed check — does the product no-gain survive the compressor draw?

Pooled 9000-obs median FoM3 per compressor seed (each = own compressor + 3-MAF-seed pooled sweep).

| compressor seed | auto-only | +product | product/auto |
|---|---|---|---|
| 41 (orig) | 2325 | 2181 | 0.94× |
| 42 | 2170 | 2393 | 1.10× |
| 43 | 2480 | 2433 | 0.98× |

**Verdict (corrected by hand 2026-06-10):** product/auto is NOT ≤ 1 across compressor seeds —
the ratio flips sign with the draw (s41 0.94×, s42 **1.10×**, s43 0.98×; mean-of-seeds 1.00×).
The strict "no-cross-gain" is therefore **not robust to the compressor draw**: the CNN's cross
effect is smaller than its compressor-seed variance (±~8%) and is consistent with **zero
systematic gain**. What IS robust: every CNN product seed (2181/2393/2433) stays well below the
L1 product 2875 (CNN/L1 = 0.76–0.85×), and the auto-only seeds (2170–2480) straddle L1 auto 2405.

*(The line previously here — "product/auto ≤ 1 across all compressor seeds ⇒ robust" — was a
hardcoded verdict written by the pre-fix driver still in memory; the generator was fixed in
commit 5f1afd9 the same day, while this run was in flight. The table above is data-derived and
correct.)*
