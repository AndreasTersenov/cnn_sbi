# PLAN — overnight rescue-menu + joint-statistics screening (GO Andreas 2026-06-11 night)

**Pre-sleep interview outcomes (Andreas):** run beside the small foreign tenants on GPUs
0/2 (polite 40% mem caps, probe before each job, back off above 12 GB foreign); matrix as
planned; doc writeups WAIT for the morning (tables + handoff only); VMIM-MLP compression of
the joint datavectors allowed but ONLY as the last thing after everything else has run, and
only if the screening evidence warrants it. Extra-test license granted (think before
burning compute). Driver parallelized across GPUs 1,0,2 (slot workers).

**Mandate:** test the §1.7 rescue items not yet checked and still relevant; test the joint
PDF; test a new "joint wavelet l1"; SCREENING rigor (1 MAF seed, 3000 obs) overnight, with
automatic escalation (3 seeds, 9000 obs — only a re-sweep, caches reusable) for arms that
qualify; morning handoff. Dimensionality kept ≤ ~3200 tonight; if posteriors look
dim-limited, the VMIM-MLP compressor route is TOMORROW's tool (Andreas).

## Scope decisions (registered)

- §1.7 item 1 (cut-then-rotate): SKIPPED — uncut rotate-back ≡ the noBNT arm (vacuous);
  whitening already measured the nontrivial rotation (1.06); the CUT version needs Andreas's
  cut protocol (not tonight's scope).
- item 2 (deep channels): DONE (0.730 / 1.082).
- item 4 (products): DONE (0.22×).
- item 3 → **A1**, item 5 → **A2**, item 6 + joint-l1 → **B/C** (below).

## Arms (screening: seed 41, 3000 obs, m=2000; pooled median FoM3)

**A1 — BNT-L1 + second moments (P7).** Append the per-scale wavelet (co)variances of the 4
BNT channels (10 per scale × 5 = 50 cols) to the bit-identical BNT-L1 800 → 850-dim.
By P7 these are an invertible repackaging of the original-basis second moments ⇒ they
restore the COMPLETE Gaussian sector exactly. The recovered fraction directly measures the
Gaussian share of what the l1 lost. Registered reading: recovered ≈ Gaussian-sector share;
no threshold (it is a measurement, not a pass/fail); ALWAYS escalated.

**A2 — BNT-L1 + 6 pairwise-union channels (survey practice / M2).** Append l1 blocks of the
6 equal-weight pair averages (κᵢ+κⱼ)/2 of the ORIGINAL bins (constructible from BNT maps;
basis-agnostic), built with the deep-block machinery (mode `unions6`, σ rows √(ΣM²σ²)):
800 + 1200 = 2000-dim. Span account predicts ~full recovery (the 6 unions span the deep
subspace richly). Registered: recovered ≥ 0.95 expected.

**B — joint one-point PDF (per scale, SNR units, fixed [−5,5] range, clamp-to-edge):**
- B1/B2: pairwise-2D, K=10 → 6 pairs × 5 scales × 100 = 3000-dim; noBNT / BNT bases.
- B3/B4: full-4D, K=5 → 625 × 5 scales = 3125-dim; noBNT / BNT bases.
Registered predictions: **B4/B3 ≈ 1** (full-4D is exactly basis-covariant, P4b — the
decisive invariance test); B2/B1 may deviate (pairwise-2D is NOT closed under mixing —
its deviation measures the pairwise approximation's basis fragility); B1 vs l1-noBNT 2405
measures whether the joint PDF is competitive as a statistic at all.

**C — joint wavelet l1 (NEW; Andreas).** Same pairwise-2D cells as B1/B2, but each cell
holds Σ(|uᵢ|+|uⱼ|)/2 over its pixels (SNR-magnitude-weighted; the strict joint
generalization of the pipeline's SNR-binned l1) instead of counts. C1 noBNT / C2 BNT,
K=10, 3000-dim. Registered: C1 vs B1 measures what l1-weighting adds over the plain PDF;
C2/C1 the basis fragility.

All arms: identical loader parameters as every campaign arm (train perms 5-6/flip/seed 1001/
batch 512; val test 0-1/noflip/2001), frozen σ tables per basis (nobnt / bnt — both exist,
GATE A1b-passed), theta/perm/patch alignment HARD-ASSERTED where concat is used; standalone
joint arms get fresh caches in the standard sweep layout. Fiducial: same 36000-obs pass.

## Escalation (automatic, registered)

After all screenings: re-sweep (3 seeds, 9000 obs — no rebuild) any arm with screening
FoM3 ≥ 0.7 × 2405 ≈ 1680, plus A1 always, plus BOTH members of a {noBNT, BNT} pair if
either qualifies (invariance ratios need matched rigor). Escalated outputs land in
`<arm>/population_sweep_full/`; the derived report quotes screening AND full numbers.

## NIGHT ADDENDA (registered as they arose)

1. **Dequantization fix** (~23:10): full4d K=5 NaN'd the MAF — diagnosed on the cache
   (median surviving feature ~4 distinct values; quasi-discrete sparse cells). Fix: counts +
   seeded U(0,1) + K=4; later extended to pair2d/jointl1 after their BNT-basis FULL sweeps
   hit the same (milder, seed-lottery) pathology. All retry arms rebuilt identically in both
   bases (matched ratios).
2. **Grid-transport test** (registered ~23:40, BEFORE running): the dequantized full4d
   screening ratio came out 0.46, far from P4b's ≈1. Hypothesis: P4b covariance holds for
   the DISTRIBUTION, but the fixed [−5,5] noise-scaled grid does not transport — in the BNT
   basis the (amplified-noise-scaled) box buries the signal in a few central cells.
   PREDICTION: rebuilding full4d with per-(channel,scale) PERCENTILE (signal-adapted) ranges
   in both bases moves the invariance ratio substantially toward 1 (registered band: ≥0.75
   supports grid-transport; ≤0.55 refutes it and leaves a genuine open problem for the
   morning). Run only if the low ratio survives the full sweeps.

## Outputs

`overnight_menu/` campaign dir: per-arm caches + logs + `OVERNIGHT_STATUS.md` (incremental,
appended after every arm) + final derived `OVERNIGHT_RESULT.md` (rescue-ladder table
extended by A1/A2; joint-statistic table with invariance ratios; escalation table) +
`HANDOFF_OVERNIGHT_2026-06-12.md` for the morning.

## Budget (measured analogs, not guesses)

Per arm: build = one TFDS pass (measured today: ~3.5–4.5 min hot cache, ~50 min cold) +
fiducial pass (~2–4 min) + screening sweep (< the measured 5–9 min full sweeps). 8 arms:
~1.5 h hot to ~8 h cold; escalation re-sweeps ~10 min/arm. Single GPU from the 0/1/2 pool,
tenant-checked at launch; sequential phases (overnight co-tenant safety).
