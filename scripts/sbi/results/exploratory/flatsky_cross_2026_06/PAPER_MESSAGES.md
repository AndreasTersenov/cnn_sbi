# Paper messages — the spine (living doc; iterate as work proceeds)

Andreas's framing (2026-06-13), with status flags. Two sessions feed this: THIS one
(analytical summary stats — L1, BNT rescue) and a SEPARATE one (CNN optimization,
HANDOFF_CNN_OPTIMIZATION.md). Status legend: **LOCKED** (result + explanation solid,
doc-backed) / **OPEN** (live work) / **VULN** (referee-exposed, needs shoring up).

---

## M1 — The learned "optimal" compressor (CNN-VMIM) does not beat the L1 norm. [VULN→to-LOCK]
Headline framing (referee-proof): *the CNN does not OUTPERFORM L1* — not "L1 wins".
Numbers (flat-local, pooled/representative): auto-only CNN best 2620 / mean 2457 vs L1 2405
= **tie**; on the product cross channel L1+product 2875 ≫ CNN+product mean 2191 (CNN cross
HURTS). So the learned compressor matches L1 on autos and is BEHIND on cross.
Explanation: CNNs are optimal only in the limit; in practice finite cosmologies + a
high-variance VMIM objective + an unstable companion flow keep them off it. Evidence the
CNN (not the data) is the limiter: the *dataset is ample* (324k patch examples / 899
cosmologies, §dataset below), so this is NOT data scarcity. **VULN:** a referee will say
"your CNN is undertrained." Defended only once the separate CNN-optimization session shows
best-effort training (architecture + NDE-flow + convergence) and the gap persists.
Backing: FLATSKY_CNN_RESULT.md; HANDOFF_CNN_OPTIMIZATION.md (the defense work).

## M2 — Designed cross-maps: convolution doesn't help, product modestly does, and we know why. [LOCKED]
Conv +4% (≈0 de-leaked), product +20% (L1). Three-leg explanation: (i) the conv map is a
lag-space re-encoding of TWO-point information; (ii) CLT-compressed to a few effective modes
on a 10° patch; (iii) Zürcher reconciliation — their cross-bin gains are
IA-self-calibration-dominated (Table 3: σ(A_IA) −104%..−430% without cross-bins) and our
forecast has NO IA, so the dominant literature channel is absent by construction. Lead with
the IA leg. Backing: FLATSKY_CROSS_RESULT.md.

## M3 — BNT: L1 contours inflate, CNN is lossless, both explained. [LOCKED]
L1-auto BNT/noBNT 0.15× (collapse); CNN 0.93× (lossless within seed scatter). Whitening test:
one fixed rotation of the nulled maps recovers the full no-BNT FoM (1.06/1.01) ⇒ no
irreducibly-joint loss; the collapse is a per-channel-shadow / no-deep-direction frame
artifact. Backing: FLATSKY_BNT_RESULT.md, BNT_THEORY_DEEP_DIVE.md (proofs P1–P7, P4c).

## M4 — Make L1 work in BNT space via map combinations, no loss. [OPEN — the live work HERE]
Rescue ladder (uncut): +deep 0.73, +deep2 1.08, +unions6 1.18, whiten 1.06. Post-cut frontier
(overnight-2 lane B, schedule-conditional, NOT yet gated): cut-BNT + linear recombinations of
kept channels ≈ 1.8× the uniform-cut no-BNT analysis; plain pairwise sums ≈ B⁻¹-reconstructed
deep. **To make it a message:** GATE C (TARP+SBC) on the winning post-cut arms + robustness
across ≥2 cut schedules. This is the next compute in THIS session. Backing: lane B in
OVERNIGHT2_RESULT.md, BNT_THEORY_DEEP_DIVE.md §1.7 item 7.

## M5 — Joint one-point statistics (A1 etc.) — keep in mind, NOT priority. [parked]
Defensible version: pairwise joint PDF of wavelet coefficients reaches ~l1+product level
from AUTO maps alone (cross-map info accessible without building cross maps). The FoM3 ~3.5k
"new best" headline does NOT survive (DPI: A1 = compression of pair2d can't add info, yet
higher FoM3 ⇒ estimator-path artifact; fiducial marginals only tied-to-~10% over l1+product;
mild Ωm/σ8 over-confidence). At most a discussion/appendix note. Backing: LANE_A_CONCLUSION.md.

---

## Cross-cutting methodological note (applies to M1, M4, M5)
FoM3 differences of ~20–30% between methods in our setup are partly NDE/estimator quality,
not physics (DPI argument; the K-trend 2874/2794/2455 decreasing with finer binning is the
clean demonstration). Robust claims are the **calibrated, marginals-level** ones. For any
"statistic A vs statistic B" ranking that matters, fix one NDE + training budget + a
convergence diagnostic and run both through it.

## Dataset facts (settles the "undertraining?" question, 2026-06-13)
Flat-local training cache: **323,640 patch examples / 899 distinct cosmologies / 360
patches per cosmology** (180 patches × 2 perms). Val: 144,000 / 400 cosmologies. Grid total
= 2500 cosmologies (CosmoGridV1). Reading: 324k patches is ample for the COMPRESSOR (feature
learning / augmentation) ⇒ data scarcity is NOT the compressor's problem. The 899 cosmologies
cap the theta-dependence the NDE can learn, but L1 and CNN face the SAME 899, and L1 does
better ⇒ the CNN's deficit is architecture/optimization/flow, not data. (Corrects an earlier
"~70k sims" mis-statement.)
