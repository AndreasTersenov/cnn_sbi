# §5.4 deep-channel test (deep2) — derived result

L1 on 4 untouched BNT maps + deep block (6ch: avg + bin4).
Registered ladders: PLAN_BNTDEEP_TEST.md (deep: >=0.8; deep2: <=0.75 refutes-at-margin / 0.75-0.95 span-supported / >=0.95 spanning).
Pooled 3-MAF 9000-obs median FoM3; same MAF/sweep machinery as all arms.

| arm | FoM3 | sigma(s8) | sigma(w0) |
|---|---|---|---|
| L1 noBNT auto | 2405 | 0.082 | 0.245 |
| L1 BNT auto | 364 | 0.176 | 0.323 |
| L1 BNT + deep (5ch) | 1854 | 0.096 | 0.256 |
| L1 BNT + deep2 | 2573 | 0.079 | 0.241 |

**recovered = (arm − BNT)/(noBNT − BNT) = 1.082**

(1-deep rung: 0.730)
**Verdict: SPANNING (>= 0.95)** — two deep directions essentially exhaust the usable signal-rich subspace.

NB mechanism test in the UNCUT information-accounting setting — not a survey recipe (deep channels would need conservative cuts; deep-dive §1.7 item 2 caveat).
