# §5.4 one-extra-deep-channel test — derived result

5-channel L1 = 4 untouched BNT maps + the plain bin average (deep channel).
Pre-registered (BNT_THEORY_DEEP_DIVE.md §5.4, before any data): recovered >= 0.8.
Pooled 3-MAF 9000-obs median FoM3; same MAF/sweep machinery as all arms.

| arm | FoM3 | sigma(s8) | sigma(w0) |
|---|---|---|---|
| L1 noBNT auto | 2405 | 0.082 | 0.245 |
| L1 BNT auto | 364 | 0.176 | 0.323 |
| L1 BNT + deep (5ch) | 1854 | 0.096 | 0.256 |

**recovered = (deep5 − BNT)/(noBNT − BNT) = 0.730**

**Verdict: PARTIAL (0.4-0.8)** — the deep direction carries a substantial share but the account is incomplete; the remainder lives in structure the single average does not expose.

NB mechanism test in the UNCUT information-accounting setting — not a survey recipe (the deep channel would need conservative cuts; deep-dive §1.7 item 2 caveat).
