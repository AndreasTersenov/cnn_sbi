---
name: 'Flat-sky cross-maps: de-leaked auto+cross (build, train, contours)'
status: active
tags:
    - experiment
    - sbi
    - cnn
    - l1
    - cross-maps
    - flat-sky
    - paper
created-at: 2026-06-08T16:45:21.09193286Z
outcome: 'RESOLVED 2026-06-10 (calibrated, multi-compressor-seed-robust): on the physically-buildable patch-local cross, L1 gains +20% (product=xi_ij; 2405->2875) while the CNN gets NO SYSTEMATIC gain (product/auto flips sign with the compressor draw 0.94/1.10/0.98, mean 1.00x — zero gain dominated by +-8% seed variance, NOT a systematic loss); every CNN product seed <= 0.85x L1 product (L1 > CNN on the explicit cross channel, draw-robust); auto-only = statistical TIE (CNN seeds 2170-2480 straddle L1 2405). ~92% of the old full-sphere cross gain was leakage. VMIM val loss product~=auto per seed => optimization-limited hypothesis lives at the RECIPE level (untested). OPEN threads: recipe-level test, principled best-seed (val-loss), BNT-cross campaign, scheduler packing (EFFICIENCY_AUDIT_2026-06-10.md). HISTORICAL OBJECTIVE (filed 2026-06-08): Replace the LEAKY full-sphere harmonic cross-maps (every cross-patch is a global functional of the whole sky; auto+cross constraining power partly UNPHYSICAL — see CROSS_MAP_LEAKAGE_FINDING.md) with PATCH-LOCAL flat-sky cross-maps, recompute stats, train L1+CNN, get CALIBRATED cosmological contours. Design+validation DONE (FLATSKY_CROSS_REDESIGN_NOTES.md S1-14): cross map = apodized-circular CONVOLUTION (Zurcher Eq.12 flat-sky analog) AND pointwise PRODUCT (its mean = xi_ij); complementary -> TEST BOTH. NO sim/dataset rebuild (cross computed on-the-fly from auto ch0-3 of TFDS grid_10deg_80px_nonoverlap180; auto-only baseline uses identical autos => clean comparison). CAUTION: do NOT reuse the old --cross-maps route wholesale (it is the bad one we dissected) — reuse only the FFT-product math + apodization window (re-validated), REWRITE the noise/SNR (per-channel, not shared auto-sigma), and compute the FFTs ON-DEVICE/batched (torch/JAX), NOT in a CPU tf.data map (starves the GPU). PRIMARY METRIC: median over typical patches of sigma(w0) + 2D(Om,s8); FoM3 reported NEVER headlined. NEXT: implement on-device augmentation (L1 --cross-op {conv,product,both}; ADD flat-sky to CNN; per-channel/per-scale noise) -> GATE A construction+throughput (bitmatch, xi_ij, benchmark augment ON vs OFF) -> GATE B cosmology-dependence (NEW, decisive) -> train matrix (auto-only / +conv / +product / +both x L1,CNN, 3 seeds) -> GATE C calibration (TARP/SBC/L-C2ST) -> contours vs auto-only AND vs full-sphere. Expect MODEST gains (cross info is large-scale, patch samples it poorly = physically correct). See FLATSKY_CROSS_BUILD_PLAN.md + HANDOFF_FLATSKY_CROSS_2026-06-08.md. Continues [[definitive-l1-vs-cnn-10deg-2026-06]].'
---

## Primary metric
median over typical patches of sigma(w0) + 2D(Om,s8) area. FoM3 reported, NEVER headlined (feedback_fom3_fragile_use_2d_areas).

## Done condition
Each arm cross-gain over auto-only is measured AND calibration-validated (TARP/SBC/L-C2ST); conv-vs-product complementarity decided; honest flat-sky gain compared to the inflated full-sphere number. Stop when contours + comparison are produced and written up.

## Guardrails
patch-local cross ONLY (never full-sphere = leakage); per-channel noise (not shared auto-sigma); never PCA L1; GPU 1 only; example-disjoint compressor/NDE split by perm; calibrate BEFORE contours; SAME auto channels across all arms; one apodized-circular convolution definition; do not relitigate the operator choice (notes S8-12).

## Loop status (DEEP-DIVE-v2.2 2026-06-12 ~10:40 UTC)
★ Theory-discussion folds (Andreas authorized): deep-dive v2.1→v2.2 — (1) GATE-C verdict
folded everywhere the stale "GATE C NOT yet run" caveat lived (header, §0 ledger row,
§1.8 TESTED block + new verdict para, §5.6) with the downgrade language; (2) NEW §1.8 tilt
picture (P4c in one paragraph: checkerboard → slanted parallelograms; ranges fix stretch
0.45→0.70, no upright grid reproduces tilt; pairwise needs other channels ⇒ only full joint
can be exact); (3) §1.7 item-2 caveat resolved by CUT-THEN-MIX (sums of cut BNT channels
reconstruct deep directions via B⁻¹ rows, inherit per-slice cleaning; span ≠ rescue
sharpening) + NEW item 7 = post-cut frontier experiment design (a/b/c/d arms, needs
Andreas's cut schedule); (4) FLATSKY_CROSS_RESULT.md GM-vs-product impact assessment
(invertible per-mode remap ⇒ identical info ceiling; branch ambiguity makes GM less
well-defined; no conclusion affected). Q2 answered from existing artifacts: full4d WAS run
(2401 = l1-auto at K=4; resolution beats joint order; occupancy/calibration worsen with
dimension) — no doc change needed beyond §1.8 already covering it.

## Loop status (CONV-THEORY-ACCOUNT 2026-06-12 ~10:00 UTC)
★ Andreas theory session: WHY-CONV-ADDS-NOTHING account assembled + folded into
FLATSKY_CROSS_RESULT.md (paper-bound section, Andreas requested): (1) conv map = lag-space
empirical cross-correlation map (pixel p = Σₓκᵢκⱼ(p−x) ≈ N_pix·ξ̂ᵢⱼ(p)) ⇒ one-point stats
of it re-encode 2-pt info; product = local ⇒ joint moments = non-Gaussian (matches +4% vs
+20% and pair2d≈l1+product); (2) CLT compression → few effective dof at 10° (also explains
seed-fragility); (3) ZÜRCHER RE-READ (authorized; arXiv:2206.01450 Table 3+§5.5): their
cross-bin gain is DOMINATED by IA self-calibration (σ(A_IA) −104%..−430% w/o cross-bins;
cosmology hit via A_IA–S8 degeneracy) — our forecast has NO IA ⇒ dominant literature
channel absent by construction; footprints 5000/14300 deg² vs our 100 deg² patches; their
Eq.12 = √âᵢ√âⱼ GEOMETRIC MEAN, ours = plain product (same global-support family, honest
def. difference noted). BNT corollary: conv rescue bounded by cov50's 0.38 (P7 invariance;
62% of loss non-Gaussian); measured BNT both=751 single-run ≈ product 637 (no rescue).
Registered-not-run: conv gain should grow with patch size (20° TFDS exists).

## Loop status (GATE-C-JOINT 2026-06-12 ~08:00 UTC)
★★ JOINT-ARM GATE C DONE (22 min, 4 arms × TARP-600/3-seed + SBC-1800, GPUs 1+2 packed
2/GPU, zero failures; bands registered BEFORE data in PLAN_GATE_C_JOINT.md; report
overnight_menu/gate_c/GATE_C_JOINT.md). VERDICTS: **pair2dq_nobnt FAIL** (tightest-tercile
TARP signed dev −0.134, SEED-ROBUST −0.092/−0.108/−0.134); jointl1q_nobnt PASS-with-caveat
(HIGH −0.080 seed-robust); pair2dq_bnt / jointl1q_bnt PASS-with-caveat (0.075/0.066). All
four arms SBC std 0.298–0.309 ⇒ ~4–6% global under-coverage. REGISTERED COMPARATIVE
DOWNGRADE TRIGGERED: over-confidence concentrates in the TIGHTEST posteriors and is the
same order as the claimed edge (σ_s8 0.072-vs-0.075, +16% FoM3 ≈ 5%/axis) ⇒ joint-stat
headline is now "reach at least l1-auto, broadly comparable to l1+product" — NEVER
"equal-or-better". P-G1 (noBNT clean) FALSIFIED = 4th falsified registered prediction
(journey material); P-G2 HOLDS. Invariance ratios less affected (same-direction errors)
but pair2d's denominator is FAIL-grade — flag carried. Directional theory point (joint
occupancy carries the product's one-point info) STANDS. Folds: OVERNIGHT_RESULT.md GATE C
section, FLATSKY_BNT_RESULT.md overnight stanza, plan adjudication, memory + index.
Paper-posture decision (third pillar vs discussion section) → Andreas, next.

## Loop status (SESSION-HANDOFF 2026-06-12)
SESSION CLOSED at Andreas's request (context 74%); next Fable 5 session entry point =
**HANDOFF_FABLE5_2026-06-12.md** (repo root; supersedes 06-11). NO jobs running; GPUs
released; everything committed+pushed through this commit. NEXT ACTIONS (priority, gates in
handoff §4): (1) GATE C (TARP+SBC) on pair2dq_nobnt/jointl1q_nobnt — REQUIRED before any
joint-stat contour enters the paper [needs go]; (2) paper assembly /paper-draft [Andreas's
call; journey-narrative memory applies; open decisions: joint-PDF as third pillar?, quote
38% Gaussian share?, two [REF]s to pin — ask before fetching]; (3) optional cheap unrun
tests (pair2da, K-scaling, BNT-adaptive-pairwise) [need go]; (4) packing benchmarks ride
the next real campaign. Session total 2026-06-11→12: whitening (1.06/1.01 full recovery),
deep-dive v1→v2→v2.1, §5.4 ladder (0.730/1.082), overnight menu (16 arms: joint stats ≈
l1+product from autos; Gaussian share 0.38; unions 1.178; grid-transport 0.45→0.70 + P4c),
corners ×2, datavector figures, 3 falsified registered predictions kept as journey material.

## Loop status (morning-session 2026-06-12 ~late-morning UTC)
★ MORNING DELIVERABLES (Andreas awake, reviewed): overnight figures (rescue_ladder /
fom3_joint_stats / invariance_ratios), typical-obs CORNERS both bases (3-seed retrain-pooled
100k samples; jaxili reload gotcha sidestepped; --replot-only path; full truth crosshairs
after Andreas caught missing horizontals), σ8-coded JOINT DATAVECTOR figures (curve grids +
native 2D heatmaps both stats × bases; jointl1 2D shows the weighting just amplifies the
joint-tail corners counts already record ⇒ explains counts≈weighting). DOC FOLD (Andreas
authorized): deep-dive v2.1 — new P4c (grids transport only diagonally; 0.45→0.70 measured),
§1.7 menu items MEASURED, §1.8 results table (marginals-first reading: pair2d σs8 0.072 ≤
l1+product 0.075, FoM3 −3% = fragility), §4.3 both registered predictions adjudicated
(invariance-as-operationalized FALSIFIED → P4c), §5.4 ladder + Gaussian-share line, new
§5.6 synthesis; FLATSKY_BNT_RESULT.md overnight stanza + figure inventory; NEW memory
project_joint_onepoint_stats_and_grid_transport. BNT corner reading logged: offsets at this
obs = ridge-slide of wide posteriors, NOT bias (l1 arms GATE-C'd; joint arms UNCALIBRATED —
GATE C required before paper use). Commits → 41745ed + this.

## Loop status (OVERNIGHT-COMPLETE 2026-06-12 ~00:10 UTC)
★★★ NIGHT DONE (16 arms total, zero unresolved failures; all numbers full-rigor 3-seed/
9000-obs unless noted). HEADLINES: (1) JOINT STATS WORK — pair2d joint PDF 2794 ≈ joint
wavelet l1 2788 vs l1-auto 2405 (+16%) ≈ l1+product 2875, NO cross-maps needed; counts ≈
l1-weighting (info = joint occupancy). (2) RESCUE MENU CLOSED — unions6 1.18 (survey
practice = full rescue, matches span story); A1 cov50: **Gaussian share of the l1 BNT loss
= 38%** (⇒ 62% non-Gaussian — sharpens F5). (3) MAIN THEORY OUTPUT: "joint PDF BNT-robust
by construction" needs the GRID-TRANSPORT qualifier — P4b covariance is of the
DISTRIBUTION; binned estimators: fixed noise-scaled grid ratio 0.45 → axis-adapted
percentile grid 0.70 (registered bands ≥0.75/≤0.55: between, toward support) → exact
transport needs SHEARED cells (B-images = parallelepipeds) impossible for axis-aligned
histograms; the CNN's first layer implements exactly that shear. (4) ENGINEERING:
count-histograms NaN the MAF on quasi-discrete sparse cells — dequantization (+U(0,1))
fixed it 3×; no dim-limitation up to 3200 ⇒ VMIM not warranted, NOT run (per registered
rule). Skipped (think-first): pair2da, K=15 scaling. All registered predictions + bands in
PLAN_OVERNIGHT_MENU.md (2 night addenda registered BEFORE runs); synthesis in
OVERNIGHT_RESULT.md; morning brief HANDOFF_OVERNIGHT_2026-06-12.md. Doc folds intentionally
left for the morning session (Andreas's call). GPUs released.

## Loop status (OVERNIGHT-MENU-LAUNCHED 2026-06-11 ~22:30 UTC)
★ OVERNIGHT SCREENING RUNNING (Andreas asleep; pre-sleep interview logged in
PLAN_OVERNIGHT_MENU.md): 8 arms on GPUs 1,0,2 (slot workers, polite 40% caps, 12 GB foreign
back-off — co-tenancy explicitly approved). ARMS: A1 = BNT-L1 + per-scale wavelet cov50
(P7 Gaussian-share measurement, always escalated); A2 = BNT-L1 + unions6 L1 blocks (survey
practice, M2); pair2d-K10 / full4d-K5 / jointl1-K10 (NEW joint wavelet l1: cells hold
Σ(|uᵢ|+|uⱼ|)/2) × {noBNT, BNT} — full4d BNT/noBNT ratio = the EXACT P4b basis-covariance
test; pair2d/jointl1 ratios = pairwise-approximation fragility. Screening 1 seed/3000 obs;
auto-escalation ≥1680 FoM3 (pairs together) → 3 seeds/9000 obs re-sweeps. Fixed [−5,5] SNR
range, K bins, clamp-to-edge; frozen σ per basis (both tables GATE A1b-passed); A1/A2
theta-bit-equality asserted against the BNT cache. Unit-tested (count conservation, np.cov
match, jointl1 totals, unions6 rows). VMIM-on-joint allowed ONLY last-if-warranted; doc
writeups WAIT for morning (tables + HANDOFF_OVERNIGHT only). Code f527173 + 0a4b1a2;
monitor armed. Skipped-as-resolved: §1.7 item 1 (rotate-back ≡ noBNT uncut; whiten was the
nontrivial rotation), item 4 (product 0.22× measured).

## Loop status (bnt+deep2-SPANNING 2026-06-11 ~22:45 UTC)
★★★ §5.4 LADDER COMPLETE — deep2 (avg + bin4, 6ch) in 13 min, all asserts PASS:
**recovered = 1.082 = the registered SPANNING branch (≥0.95).** FoM3 2573 (vs noBNT 2405);
σ_s8 0.079 (vs 0.082), σ_w0 0.241 (vs 0.245) — EVERY marginal at or better than noBNT. ⇒
The 1-deep residual (27%) WAS among-deep-kernel tomographic structure: the second
depth-distinct direction retrieves all of it. SPAN CURVE FINAL: 0 deep directions → 0.00
(the 0.15× collapse), 1 → 0.730, 2 → 1.082, orthonormal-4 (whiten) → 1.06. Per-channel-
accessible info saturates at ~2 depth-distinct deep directions for these params. >1 values
recur across two independent frames ⇒ the standard per-bin frame is itself a mildly
suboptimal one-point direction sampling (stated, not over-read). PRACTICAL (uncut): BNT + 2
fixed channels = complete per-channel recovery, nulled maps untouched; cut-analysis caveat
carried. σ check: bin4 row = 2× avg row exactly (noise-halving of the average ✓). Folded
into deep-dive (§0 ×2, ledger, §1.7 item 2, §5.3, §5.4 + span-curve table), FLATSKY_BNT_
RESULT.md, paper drafts I+II, figure (5-bar ladder), memory. The §5 journey arc now: F4
falsified → sign-structure dead → 1-deep below registration → 2-deep lands the registered
SPANNING branch — the account CONVERGED under three corrections.

## Loop status (bnt+deep-§5.4-RESULT 2026-06-11 ~22:00 UTC)
★★ §5.4 DONE in 11 min (build 198s — TFDS hot — sweep 275s, zero failures, all alignment
asserts PASS): **recovered = 0.730 — BELOW the registered 0.8 ⇒ PARTIAL.** deep5 FoM3 1854
(noBNT 2405, BNT 364); σ_s8 0.176→0.096 (vs 0.082, ~85% of damage undone); σ_w0 ≈ restored
(0.256 vs 0.245). READING: deep direction = DOMINANT carrier (one fixed channel undoes 3/4
of the collapse) but the single-direction STRONG form is REFUTED — residual ~27% =
tomographic structure among the 4 deep kernels (different depths; one average can't carry
it). Account refined to SPAN-CALIBRATED: per-channel info scales with spanned signal-rich
subspace — 0 directions→0.15×, 1→0.73, spanning set (whiten/original)→1.0. Caveat biasing
0.73 DOWN: deep channel's 40 bins over a wide heavy-tailed range ([−12,14] SNR) = coarser
core binning. Registered-not-run next rung: 2nd depth-distinct deep channel (predicted
strictly between 0.73 and 1). Folded into deep-dive §0/§1.7/§5.3/§5.4 + ledger, FLATSKY_BNT_
RESULT.md, paper draft Part I+II, memory, figure (4th bar). Honesty chain now THREE falsified
/sub-threshold predictions kept as journey material (F4 LOW-MID → full; sign-structure → dead
on geometry; §5.4 ≥0.8 → 0.73).

## Loop status (bnt+deep-§5.4-LAUNCHED 2026-06-11 ~21:45 UTC)
★ §5.4 TEST RUNNING (Andreas GO): run_flatsky_bntdeep_campaign.py detached GPU 1 (sole
tenant), monitor armed. PRE-REGISTERED: recovered = (deep5−BNT)/(noBNT−BNT) ≥ 0.8 / 0.4–0.8
partial / <0.4 refutes §5.3. Implementation = per-channel block CONCAT (no 5-ch plumbing):
x = [cached flat_local_none_bnt 800 cols (bit-identical to the measured 0.15× arm) | fresh
deep-channel 200 cols]; deep σ(s) = ¼√(Σσⱼ²) derived EXACTLY from the verified no-BNT table;
loader params mirror the BNT build (train/5-6/flip/1001/512, val/test/0-1/noflip/2001);
theta + fiducial perm/patch bit-equality HARD-ASSERTED. New: mix modes 'deep'(1×4 avg)/
'bnt_deep'(5×4) + mode-aware n_built_channels (unit-checked: avg exact, row sums 1,0,0,0,1).
Build streaming at ~3200 patches/s (TFDS hot in page cache after whiten campaign — vs 169/s
cold). Code committed 60c9b61 + PLAN_BNTDEEP_TEST.md. Bin-count remarks (≥4-bins folklore
derived: 2-param kernel ⇒ 2 conditions ⇒ 3 bins/row, N−2 nulled maps; row 2 only partially
nulled) folded into deep-dive §1.2/F3 (f2a9e3c); Q-vs-B⁻¹ + survey-workflow clarifications
(7258411).

## Loop status (whitening-vs-Binv + survey-workflow clarifications 2026-06-11 ~16:30 UTC)
Andreas Q&A round 2: (1) Q vs B⁻¹ — clarified in doc §1.6: ALL noise-whiteners of the nulled
maps form the family W = O·B⁻¹ (net transform = rotation); B⁻¹ is the O=I member (would have
vacuously re-measured the no-BNT arm); Q is the symmetric member landing on a genuinely
rotated frame ⇒ the test is falsifiable, and the null→cut→invert→measure pipeline Andreas
says is PROPOSED in the literature [REF needed] = our §1.7 item 1 — our results are its
quantitative justification for higher-order stats. (2) Survey-purpose check (his ℓ↔k-alignment
framing adopted, §1.7 item 0): joint PDF does NOT defeat BNT — cuts in BNT space then joint
histogram of the CUT NULLED MAPS directly (frame-indifferent, never leave BNT space); CNN
likewise; L1-rotate-back fine post-cut. CASUALTY flagged honestly: the §5.4 +1-deep-channel
idea is a mechanism test only — in a cut analysis the deep map reintroduces ℓ↔k leakage and
must be conservatively cut, eroding the gain.

## Loop status (deep-dive-v2 2026-06-11 ~16:00 UTC)
★ DEEP-DIVE v2 (Andreas review response): single-scale throughout (wavelets dropped from all
explanations), new plain-language layer §1 (cloud/shadows/CT-scan for directions; whitening
explained simply; joint-PDF concrete + COMPUTABLE verdict: pairwise-2D 1350-d < L1-both
3200-d, GPU-trivial, no covariance obstacle, ~3 h mini-campaign IF green-lit), walked proofs,
practical 6-item rescue menu (§1.7) headed by cut-then-recombine + NEW cheap idea: append ONE
deep channel (bin average) to the 4 untouched nulled maps. ★ CORRECTNESS RE-PASS FOUND v1
POST-MORTEM WRONG: Q rows 2–4 are NOT same-sign (row4 = bin3−bin4 diff) and rows 3–4 lie
95–99% INSIDE the nulled span ⇒ sign-structure story dead; alignment-angle story
insufficient. SURVIVING ACCOUNT (with new F5b slice bound, Cauchy–Schwarz): BNT = the unique
frame with NO deep direction (1 shallow bin-1 + 3 thin slices; deep non-Gaussian common
structure removed from every channel); Q recovers via leading row ≈ deep common mode (70%
outside nulled span); F3 toy was 'too kind' precisely because ITS nulled basis kept the deep
direction. Full 3-stage post-mortem chain kept in §5 as journey material. ★ NEW P7 (proved):
auto+cross 2nd moments transform invertibly (Ĉ'=BĈBᵀ) ⇒ 2-pt-with-crosses EXACTLY
BNT-invariant — PREDICTS the reported literature result Andreas relayed (autos alone
unprotected); practical corollary: +10 numbers restore the Gaussian sector free. ★ REGISTERED
PREDICTION §5.4 (not run): 5-channel L1 (4 nulled + bin-average) ⇒ recovery ≥0.8 — would be a
practical rescue PRESERVING per-slice cuts. Ripples: paper draft Part I+II rewritten (incl.
P7 section), FLATSKY_BNT_RESULT.md whitening §, memory v2. Cap lifted by Andreas (554 lines).

## Loop status (WHITEN-RESULT 2026-06-11 ~15:00 UTC)
★★★ WHITENING TEST DONE (2h22m, zero failures, ahead of ETA) — **FULL RECOVERY**: recovered
fraction (whiten−BNT)/(noBNT−BNT) = **1.06 (L1 auto: 2405→2524 vs BNT 364) / 1.01 (product:
2875→2897 vs 637)**, complete marginal-by-marginal (whiten σ_s8 0.080/0.075 ≈ noBNT
0.082/0.075; σ_w0 0.239/0.233 ≈ 0.245/0.238). ⇒ The L1 BNT collapse has irreducibly-joint
share ≈ 0 — it is ENTIRELY a FRAME artifact: the nulling rows are signal-poor,
signed-differencing directions (cancel coherent non-Gaussian content; F5 bites along them);
one fixed orthogonal rotation Q of identical mixedness (same-sign rows) hands everything
back. NB: noise correlation is invisible to marginals & amplification absorbed by SNR
re-freeze ⇒ neither was ever a complete mechanism. ★ HONESTY: the deep-dive's pre-registered
F4 prediction (LOW-to-MID) was FALSIFIED — post-mortem written into L5.2 (sign-structure
resolution); F3's "Gaussian one-point predicts no collapse" CONFIRMED from the measurement
side. Third-pillar note: joint-PDF datavector loses its BNT-rescue motivation (fixed rotation
suffices); remaining case = level-3-vs-level-1 info (product's +20%). Folded into:
FLATSKY_BNT_RESULT.md (new section + inventories), deep-dive L0/F4/L5.2, paper draft Part
I+II, memory, figure fom3_whiten_decomposition (whiten_campaign_figure.py). PACKING
DECISIONS (Andreas): benchmarks DEFERRED to next campaign's first phase; footgun fixes land
now; 3-pack only when run (PLAN_PACKING_BENCHMARKS.md).

## Loop status (theory-deep-dive 2026-06-11 ~14:10 UTC)
★ BNT THEORY DEEP-DIVE DELIVERED (Andreas-interviewed, plan signed off, no numerics, ≤400
lines): BNT_THEORY_DEEP_DIVE.md is now the canonical treatment — claims ledger
(PROVED/MEASURED/MECHANISM), L2 propositions WITH proofs (posterior/MI invariance; CNN class
closure incl. preprocessing; P4a configuration-preserving info flow = the exact "where does it
go" answer; P4b joint-one-point envelope; strict hierarchy w/ witnesses; P6 Gaussian-sector
l1⟺variance), L3 closed-form Fisher analysis. ★ HEADLINE DERIVED RESULT (F3, "the trap"): in
the honest zero-lag Gaussian toy with perfect nulling, per-channel variances become MORE
efficient in the nulled basis (I_diag^BNT ≈ I_full > I_diag^orig, closed form) — the
"suppressed S/N + correlated noise" story predicts the OPPOSITE of the measurement ⇒ the
0.15× collapse must live in: F5 Gaussianization lemma (mixing contracts standardized
cumulants, proved via ℓ^p-monotonicity; signed nulling rows cancel odd cumulants), F3.4
residual joint response (B nulls kernels not covariances), F3.5 SNR-grid flattening. F4
pre-registers the whitening reading (predict LOW-to-MID recovered; Q is still a mixing so F5
damage survives whitening). F6 anisotropy adjudicated from artifacts: w0 is substantially
PRIOR-CAPPED (max possible inflation 1.9× vs measured 1.32×; σ8 3.5× vs 2.15×) — all three
params lose 61–78% of their available room; dumbbell ordering largely reflects room-to-lose +
σ8's non-Gaussian-amplitude exposure. L4: union-catalog identity PROVED (count-weighted combo,
same noise realizations ⇒ no data-access limitation); constructive Cramér–Wold: pairwise
unions provably COMPLETE at order 2, INCOMPLETE at order 3 (k−1 weight ratios for order k;
noise-bias-free for k≥3); M3 joint-PDF datavector design (~6.5k numbers, BNT-robust by P4b).
PAPER DOC REVISED: Parts I–II mechanism corrected to the derived chain; Parts III–IV
superseded/removed (git history retains). tomo4_bnt_v1 exact constants: noise corr ρ12=−0.707
ρ23=−0.740 ρ34=−0.548; BB^T eigvals 0.088→5.60 (cond 63). Whiten campaign in flight
(both_build DONE 3030s, l1_arms running, monitor armed).

## Loop status (session-handoff 2026-06-11 ~13:40 UTC)
SESSION HANDOFF to the next Fable 5 session: HANDOFF_FABLE5_2026-06-11.md (repo root, the new
entry point — supersedes the 06-10 handoff). ★ RUNNING: run_flatsky_whiten_campaign.py (pid
3108884, GPU 1, launched ~12:45) — per-channel L1 in Q=(BB^T)^-1/2 B (orthogonal, verified) →
whiten_campaign/WHITEN_RESULT.md with recovered fraction (whiten−BNT)/(noBNT−BNT) decomposing
the 0.15× L1 collapse into noise-geometry vs irreducibly-joint; ETA ~15:30–16:30 UTC; THE OLD
SESSION'S MONITORS ARE DEAD — next session must check driver.out itself. ★ PRIORITY 2 (Andreas,
explicit): the theory explanation (Parts I–IV of PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md) was NOT
thorough enough for him — deepen it (honest 2-bin Gaussian Fisher diag-vs-joint, σ8-anisotropy
mechanism, possible numerical mini-demos, consolidate the four accreted parts into one layered
doc; interview him on what felt thin first). Backlog: Tier-1 scheduler packing (benchmarks in
EFFICIENCY_AUDIT never run), paper assembly (/paper-draft), joint-PDF third-pillar idea (needs
explicit go). Both pillars otherwise COMPLETE (results+calibration+figures+drafts). Session
total: 20 commits 5f1afd9→b551586 + this, all pushed.

## Loop status (bnt-figure-kit 2026-06-11 ~10:30 UTC)
★ BNT FIGURE KIT COMPLETE + one NEW DIAGNOSTIC FINDING. Figures (bnt_campaign/figures/):
fom3_bnt_inflation (headline log bars), sigma_bnt_dumbbell (σ8 hit hardest, w0 mildest —
parameter-anisotropic damage), sbc_bnt_ranks (CNN hugs center=conservative, L1 edges raised=
over-confident — visual match to the numeric SBC), lc2st_bnt_cnn, tarp_bnt_colored_dim{3,6}
(RESTYLED per Andreas: campaign colors CNN-blues/L1-oranges, 16–84% bands at alpha 0.40 — bands
intrinsically tight), corner overlays (delivered 06-11 am), and NEW datavectors_bnt_vs_nobnt_s8
[_relative]: σ8-quantile-binned L1 'both' datavectors in both bases from the on-disk training
caches. ★ The RELATIVE version visualizes the mechanism: under BNT the auto+conv blocks lose
nearly ALL σ8 response while the PRODUCT block retains the most — per-channel collapse + partial
cross-channel rescue in one figure (paper-grade diagnostic; backs the Part II 'where the
information goes' argument empirically). Scripts: bnt_campaign_figures.py, plot_tarp_bnt_
colored.py, plot_bnt_datavectors.py. FLATSKY_BNT_RESULT.md gained a figure inventory. Pillar 2:
result + calibration + figures + paper drafts ALL complete and committed.

## Loop status (bnt-gate-c 2026-06-11 08:40 UTC)
★ BNT GATE C DONE (corners→overlays→TARP→SBC→L-C2ST, ~5.5 h, zero job failures): PASS WITH
CAVEATS, HEADLINE-SAFE. Deviations ≈5–10% credible-width vs 90%(L1)/≤10%(CNN) effects. L1-BNT
mildly OVER-confident (SBC std 0.295–0.304) ⇒ true inflation ≥ measured ⇒ predictions 1–2
protected. CNN: SBC mildly conservative; TARP cnn-auto tightest tercile −0.068 (mild
over-conf), others conservative; L-C2ST auto 13% (mild, self-test powered), product 40% (LOCAL
miscalib at fiducial — real flag). BNT space measurably harder to calibrate than original basis
(no-BNT arms were ≤0.037/0%). Paper phrasing: losslessness from the AUTO arm (0.93×); product
0.88× with explicit caveat. Corner overlays delivered + committed (cf25cb6); GATE_C_BNT.md has
the full derived tables (TARP parser fixed for ecp_bootstrap). Paper docs: FLATSKY_BNT_RESULT
caveats updated; PAPER_BNT_MAIN_AND_APPENDIX_DRAFT.md (main-text + formal appendix + Part III
joint-PDF/Cramér–Wold/catalog-cross-maps discussion). PILLAR 2 NOW COMPLETE: result +
calibration + figures + paper drafts.

## Loop status (BNT-RESULT 2026-06-10 23:55 UTC)
★★★ BNT CAMPAIGN DONE (3.7 h, zero failures) — ALL THREE PREDICTIONS HOLD. Inflation
FoM3_BNT/noBNT: L1 auto 0.15× (2405→364; σ_s8 0.082→0.176 DOUBLES, σ_Om +69%) | L1 +product
0.22× (637; every marginal better than L1-auto-BNT — explicit cross channel partially recovers)
| CNN auto 0.94/1.00/0.86 → mean 0.93× | CNN product → 0.88× (marginals ≤3%; within/near the
±8% comp-seed band). ⇒ BNT inflation is an ANALYSIS artifact of per-channel statistics; the
channel-mixing CNN is BNT-invariant ⇒ BNT empirically LOSSLESS. PLAIN CNN sufficed (no advanced
arch needed at 10°, unlike the 20° campaign's 0.85×). PILLARS UNITED: CNN can't exploit the
explicit cross (pillar 1) but doesn't need it (pillar 2); L1 gains +20% from it no-BNT and loses
85% without it under BNT — cross-maps are a device FOR per-channel statistics, both halves shown.
Writeup FLATSKY_BNT_RESULT.md (root) + memory project_flatsky_bnt_losslessness. REMAINING for
paper-grade: GATE C (TARP/SBC + L-C2ST-CNN) on the BNT arms — runners exist, ~2 h on 2 GPUs.

## Loop status (bnt-campaign-LAUNCHED 2026-06-10 20:05 UTC)
★ BNT CAMPAIGN RUNNING (run_flatsky_bnt_campaign.py pid 655563, GPUs 1+2, detached; monitor
armed). P0 sigma freeze --bnt DONE in 30 s, GATE A1b ALL PASS — empirical BNT auto noise matches
the analytic mixing prediction sqrt(sum B_ij^2)=(1.000,1.414,1.820,1.621) to 3 digits (all 4 bins
collapse onto identical white ratios after the factor); conv/product sigma depart from white
(L1-dist 0.470/0.228); inter-bin corr +0.0026. flatsky_cross_noise_sigma_bnt.npz written. P1
both-BNT build running (the loader-bound pass). Then: L1 arm slices + 6 CNN BNT compressors →
fidsumms → 8 jit sweeps → BNT_CAMPAIGN_RESULT.md (derived prediction-ladder verdict). ETA ~5-6 h
(≈ 01:00-02:00 UTC). Andreas gave GO 2026-06-10 ~20:00.

## Loop status (recipe-verdict 2026-06-10 16:45 UTC)
★ RECIPE-LEVEL CHECK DONE (160k + val-batches 16, seeds 42/43, paired vs 80k): THE HEAVIER
RECIPE DOES NOTHING — 160k/80k per seed: auto 1.08/0.97, product 1.00/1.01 (mean auto 1.02×,
product 1.00×); CNN/L1(product) 0.83–0.85× UNCHANGED from 80k. The optimization-limited
hypothesis is now falsified at BOTH the seed level (multiseed) and the recipe level; remaining
untested rung = ARCHITECTURE (prior 20° BNT campaign needed the 'advanced' arch). Bundled-change
caveat moot (nothing moved ⇒ no ablation needed). METHODOLOGY footnote: 160k honest 16-batch val
losses are ~0.15–0.2 nats WORSE than the 80k single-batch 'best' values — the old criterion
selected on noise — yet downstream FoM3 unchanged ⇒ checkpoint-selection noise never decisive.
Best-val steps 54k/94k/66k/144k (optimum drifts past 80k without FoM3 movement = flat plateau,
data-limited). Auto-written verdict was DERIVED this time (fixed generator). Writeup: derived
recipe line added to FLATSKY_CNN_RESULT.md robustness section; memory updated. BNT contingency
ladder: rung (a) 160k-recipe DEPRIORITIZED (doesn't transfer guaranteed, but adds nothing in
no-BNT space) → if CNN-BNT inflates, go straight to advanced arch. GPUs 1+2 FREE. BNT campaign
launch ARMED, awaiting Andreas's go.

## Loop status (bnt-campaign-ready 2026-06-10 ~16:15)
★ BNT CAMPAIGN ORCHESTRATOR READY (run_flatsky_bnt_campaign.py, dry-run validated): P0 sigma
freeze --bnt (skipped if table exists) → P1 L1 both-BNT build (solo) → P2 L1 {none,product} BNT
slices + P3 CNN {none,product}×s{41,42,43} BNT compressors (RECIPE-MATCHED to no-BNT baseline:
plain/80k/val-batches-1 — inflation ratios must compare like-trained arms) → fidsumms (CNN ×6 +
L1 precompute --bnt + per-arm slice) → 8 jit sweeps → BNT_CAMPAIGN_RESULT.md with DERIVED
inflation table (no-BNT refs read from disk: L1 2405/2875, CNN per-seed multiseed medians) +
derived prediction-ladder verdict (L1-auto <0.9 / L1-product > L1-auto / CNN >0.9, contingency
note if CNN inflates). precompute_fiducial_both_datavectors.py parameterized (--bnt/--both-cache/
--sigma/--out). Est. ~5-6 h on 2 GPUs (post-jit). AWAITING Andreas's GO + free GPUs (recipe
check still running).

## Loop status (bnt-wiring-built 2026-06-10 ~15:45)
★ BNT WIRING BUILT + GATE-A VALIDATED (CPU, all 8 op×bnt combos PASS; matrix det=1.0 exactly —
unit-determinant lower-triangular). Single source of truth: flatsky_cross.apply_bnt_{np,torch,jax}
+ bnt= switch on all three build_channels_* (BNT'd autos feed BOTH auto and cross channels).
CNN: --flatsky-bnt (npe_cnn + fidsumm) → RMS estimator + jitted transform, tomo4-validated,
mutually exclusive with legacy --apply-bnt, save-path _bnt suffix, CONDITIONAL flatsky_bnt cache
fingerprint key (old caches stay valid; BNT can't hit a no-BNT cache). L1: --apply-bnt now works
on flat_local (was hard error) → SNR calibration + dataset passes + obs L1; cache dir _bnt;
both-cache space-mixing tripwire. freeze_flatsky_cross_noise.py --bnt → _bnt.npz with bnt=True
key; select_frozen_sigma HARD-ERRORS on table/arm BNT mismatch (fences the May wrong-σ failure
mode). GATE A1b auto-vs-white uses the analytic BNT reference (per-bin sqrt(sum B_ij^2); BNT of
independent whites stays white PER MAP — correlation is between bins). REMAINING before campaign
launch: BNT noise-freeze run (1 GPU job when recipe check frees a slot) → campaign orchestrator →
Andreas's go. Recipe check (160k) still in compressor phase.

## Loop status (recipe-check-launched + BNT-scoped 2026-06-10 ~15:00)
Andreas: START recipe test, SKIP L1 per-seed retrain, BNT = PAPER PILLAR 2. ★ LAUNCHED
run_recipe_160k_check.py (pid 3977580, GPUs 1+2, idle co-tenants): {none,product} × s{42,43} at
160k steps + --compressor-val-batches 16, paired vs the 80k multiseed numbers → multiseed_160k/
RECIPE_160K_CHECK.md (~2 h; monitor armed). Tests the RECIPE-level optimization-limited
hypothesis AND calibrates the BNT contingency ladder. ★ BNT PLAN REWRITTEN with Andreas's
framing (NEXT_THREADS_PLAN §B): prediction ladder = L1-auto inflates / L1+product inflates less /
CNN ~no inflation ⇒ BNT lossless for channel-mixing compressors. Inflation ratio FoM3_BNT/
FoM3_noBNT vs existing arms. Grounding: prior 20° campaign (advanced arch, 120k, 5 seeds)
recovered 0.85× FoM3 (+3.7% std_sum) — near-lossless but needed the BIGGER compressor; plain CNN
may inflate here without falsifying losslessness → contingency ladder (160k recipe → advanced
arch → discuss). Implementation: tomo4_bnt_v1 matrix from bnt_utils (bin-dependent only, carries
to 10°), JAX applier behind --flatsky-bnt in make_flat_cross_transform + np oracle + L1 torch
twin, GATE A extended, L1 noise σ RE-FROZEN through BNT (noise-only realizations must be BNT'd —
correlated post-BNT noise is the point). 4 arms (L1/CNN × auto/auto+product, BNT) + 2 extra CNN
compressor seeds for the headline ratio (multiseed lesson). AWAITING Andreas sign-off on §B
before building.

## Loop status (jit-sampling-adopted 2026-06-10 ~14:00)
★ SAMPLING JIT MEASURED + ADOPTED. bench_sample_jit.py (GPU 2): eager 183 ms/obs → jit 1.05 ms/obs
(174×); vmap unnecessary. Bit-identity fails at TF32 kernel level (max|Δ| 3.4e-3, same keys/u-draws)
⇒ adoption gate = full-arm rerun: validate_jit_sweep.py re-derived none_s42's 9000-obs pooled
median in 49 s (vs ~4100–4800 s eager): FoM3 −0.39%, σ ±0.2% — 10× inside seed scatter
(jit_validation.json). Wired as DEFAULT into population_sweep_flatsky.py (--sample-eager = legacy
bit-exact path). Sweeps now NDE-training-bound ~30 min/arm (was ~100). Keys-not-bits is the new
reproducibility contract for sweeps (flagged to Andreas). Packing benchmarks (NDE-training
co-residency, the post-jit bottleneck) deferred — GPUs 0/1 have ACTIVE foreign tenants right now;
run before the BNT campaign launch. Audit fix batches A+B all landed; scheduler Tier-1 next.

## Loop status (multiseed-verdict 2026-06-10 ~13:30)
★★ MULTI-COMPRESSOR-SEED CHECK DONE (277 min, 4 sweeps n=9000 each). Pooled 3-MAF 9000-obs median
FoM3 — auto: s41 2325 / s42 2170 / s43 2480; product: 2181 / 2393 / 2433 ⇒ product/auto = 0.94 /
**1.10** / 0.98 (mean-of-seeds 1.00×). VERDICT (mixed branch): the strict "CNN gains NOTHING" is
NOT compressor-seed-robust — the cross effect FLIPS SIGN with the draw; correct claim = ZERO
SYSTEMATIC gain, dominated by compressor-seed variance (±~8%). ROBUST facts: every CNN product
seed ≤ 0.85× L1 product 2875 (L1 +20% stands; L1 > CNN on the explicit cross channel across all
draws); CNN auto seeds STRADDLE L1 auto 2405 ⇒ auto-only = statistical tie. VMIM val loss product
≈ auto per seed (Δ≲0.02 nats) ⇒ compressor objective sees no extra MI in the product channel at
this recipe ⇒ optimization-limited hypothesis moves to the RECIPE level (untested), seed-level
rescue falsified. Writeup: FLATSKY_CNN_RESULT.md gains a derived "Robustness — compressor seed"
section (generator-emitted, consolidate_cnn_vs_l1.py); MULTISEED_COMPRESSOR_CHECK.md verdict
hand-corrected (the in-flight driver predated the 5f1afd9 hardcoded-verdict fix and wrote the
WRONG verdict against its own 1.10× table — live demonstration of the bug). Memory
project_flatsky_cnn_no_cross_gain rewritten. NB the A2 "~20% on-device cross cost" note below is
WRONG (run-level medians: product/both at parity or faster than none; 75-vs-93 was ambient load).
NEXT: bench_sample_jit on freed GPUs → packing benchmarks → scheduler plan (EFFICIENCY_AUDIT_
2026-06-10.md); audit fix batches A (5f1afd9) + B (dabda3a) + GPU-policy update (c0ae139) landed.

## Loop status (audit+fixes 2026-06-10 ~12:15)
★ PIPELINE AUDIT DONE (4 parallel subagents, read-only): PUBLISHED NUMBERS STAND — disjointness
CLEAN (metadata perm-filter + runtime audit; artifact row counts verified), RMS-whitening CLEAN
(bit-identical via deterministic recomputation + G1 ≤1.9e-4), aggregation CLEAN (n=9000/9000 all 8
arms, best_val genuine, preprocessing consistent). All findings forward-looking: dead NaN guard in
train_with_nan_retry (hasattr on dict — never fires); make_headline_corner.py conditioned on train
row 0 not the obs (latent, never produced output); HARDCODED VERDICTS in run_multiseed_compressor_
check.py + consolidate_cnn_vs_l1.py + plot_best_seed.py title; --decay-steps silently inert in
jaxili (LR const 1e-4, symmetric); "3 MAF seeds" vary ONLY the data split (flow init fixed at 42);
best_val selection = single random 128-batch (noisy); sweeps are HOST-DISPATCH-BOUND (jaxili
.sample un-jitted, ~600 dispatches/call; dim-3200 only ~25% slower than dim-10) ⇒ jit fix ~10
lines. Full report: cnn_phase/../PIPELINE_AUDIT_2026-06-10.md. BATCH-A FIXES COMMITTED 5f1afd9
(derived verdicts, wrong-obs fix, --compressor-val-batches, channel_scale+effective-policy in
cache meta, truth key). BATCH B (population_sweep_flatsky.py + npe_l1norm imports) DEFERRED until
the multiseed driver exits — pending product sweeps execute those files from disk. bench_sample_
jit.py ready (bit-identity gate + timings, run on GPU 1 when free). NEXT_THREADS_PLAN_2026-06-10.md
drafted (best-seed-by-val-loss, BNT design: noise→BNT→cross-build→whiten, batch-B+jit). MULTISEED
EARLY READ: auto-only compressor seeds 2325/2170/2480 STRADDLE L1 2405 (best +3%, worst −10%);
product-vs-none compressor VMIM val loss ≈ EQUAL per seed (−10.76≈−10.75 s42, −10.80≈−10.82 s43)
⇒ product channel adds no measurable MI at the compressor objective. Product sweeps land ~13:40.

## Loop status (fable5-handoff 2026-06-10)
SESSION HANDOFF to Fable 5 written: HANDOFF_FABLE5_2026-06-10.md (repo root, the new entry point) +
FABLE5_FIRST_PROMPT.md (copy-paste first prompt). ★ LIVE QUESTION (Andreas 2026-06-10): is the CNN
no-cross-gain OPTIMIZATION-LIMITED, not a real method difference? His point: CNN gets bins as channels
⇒ can learn cross-correlations implicitly ⇒ explicit cross redundant for CNN; cross-map trick is for
per-channel methods (L1). NOT a 10-d bottleneck (10-d enough for 6 params) — he reads it as compressor
TRAINING INEFFICIENCY, and notes BEST seed (2620) > L1 auto (2405). Data nuance: CNN auto TIES L1 auto
(not beats), big per-seed scatter (auto 2620/2364/2387, product 2225/2331/2017) ⇒ optimizer-into-
optima not capacity wall; but that scatter is MAF-seed (1 compressor), best-vs-L1 is best-vs-pooled
single-obs (suggestive not clean). RUNNING NOW: run_multiseed_compressor_check.py (compressor seeds
42,43 × {product,none}, GPU 1+2, ~4h) → does a well-trained COMPRESSOR lift product toward/over L1?
→ cnn_phase/multiseed/MULTISEED_COMPRESSOR_CHECK.md. NEXT-SESSION TODO: (1) interpret it + reframe
FLATSKY_CNN_RESULT.md/memory IF supported; (2) principled best-seed (val-loss, not post-hoc FoM3;
L1 per-seed needs retrain — 2000-d reload truncates); (3) BNT for flat-sky cross (scope first); (4)
bug/inefficiency audit (fan out subagents). Commits through cf96f25→(this). Branch pushed.

## Loop status (cnn-best-seed 2026-06-10)
★ CNN BEST-SINGLE-SEED (un-pooled) CHECK DONE 2026-06-10. The no-cross-gain is NOT a pool-haircut
artifact: reloaded the per-seed MAF checkpoints (CNN 10-d reloads bit-exact; L1 2000-d truncates →
can't), sampled each MAF seed separately at the typical obs (perm16/patch23). Best single seed FoM3:
auto 2620 | conv 2491 (0.95×) | product 2331 (0.89×) | both 2475 (0.94×) — every cross arm STILL
≤ auto-only. Auto-only tie holds (CNN best-seed 2620 ≈ L1 pooled 2487; fair pooled-vs-pooled 2325 vs
2405). Even CNN's best-seed product is wider than L1 pooled. (NB: MAF seeds, not compressor seeds —
still 1 compressor seed; the multi-COMPRESSOR-seed check remains the open robustness follow-up.)
Artifacts: cnn_phase/best_seed/{CNN_BEST_SEED.md, per_seed.json, fom3_best_seed.*, corner_best_seed_
product.*, corner_best_seed_vs_l1_<arm>.*}; scripts cnn_per_seed_best.py + plot_best_seed.py. Baked
into FLATSKY_CNN_RESULT.md (§Robustness) + GATE_C_INTERPRETATION.md. Commits d4884b3→cf96f25, pushed.

## Loop status (cnn-overnight 2026-06-09 22:10)
AUTONOMOUS OVERNIGHT PIPELINE LAUNCHED (overnight_cnn_pipeline.sh, detached pid 1518292). Waits for
the population sweep, then runs: headline consolidate → SBC(cnn) → L-C2ST(cnn) → representative
corners(3-seed,4 arms) → final consolidate. Logs+PASS/FAIL in cnn_phase/STATUS_OVERNIGHT.md;
deliverables: FLATSKY_CNN_RESULT.md (root, L1-vs-CNN table), cnn_phase/figs/ (overlays+bars),
cnn_phase/gate_c/{tarp_drp,sbc,lc2st}/.
STATE: matrix DONE (4 arms plain-CNN seed41 80k; `both` NaN'd → fixed w/ --compressor-grad-clip 1.0,
best_val@38k). Fiducial summaries DONE (9000/arm, all 4 G1 PASS max|Δ|≤1.4e-4). GATE C TARP DONE
(product/none/both calibrated-or-conservative; conv-HIGH mildly OVER-CONFIDENT −0.068, least
important arm). SBC(cnn) TESTED: means≈0.5 no bias, std 0.273-0.281 (<0.289 ⇒ mildly CONSERVATIVE/
over-cover, matches TARP), KS flags mild non-uniformity Om/s8 (safe direction). Population sweep
RUNNING (single-wave 4-GPU; ETA ~22:30) → the 9000-obs median FoM3 table.
SINGLE-OBS PREVIEW (perm0/patch90, CNN product 1-seed FoM3 2549 vs L1 2676; overlay = comparable
contours): de-leaked CNN≈L1 (NOT the leaky full-sphere CNN≫L1 ~17k vs 8.5k). Headline holds: the
big CNN cross advantage was mostly leakage. New scripts: npe_cnn_nbody_tomo.py --cnn-map-route
flat_local + --cross-op + --compressor-grad-clip; build_fiducial_summaries_cnn.py flat_local;
gate_a_flat_cross_cnn.py; run_flatsky_cnn_{matrix,fiducial_summaries,gate_c_tarp,gate_c_lc2st,
population_sweep,repr_corners}.py; compute_sbc_from_tarp_dumps_cnn.py; cnn_representative_corners.py;
quick_single_obs_cnn.py; consolidate_cnn_vs_l1.py. NONE committed. NEXT (morning): read
STATUS_OVERNIGHT.md + FLATSKY_CNN_RESULT.md; verify L-C2ST/repr corners PASSed; write memory +
final writeup; git stage by path.

## Loop status (cnn-live)
CNN PHASE STARTED 2026-06-09 (session 3). Decisions LOCKED w/ Andreas: plain CNN (conv 64,128,256/
dense 256/dim 10, NO BatchNorm); 1 compressor/arm seed-41 + 3-MAF-seed pooling downstream (mirrors
Phase D, symmetric w/ L1 — removes NDE confound via the COMMON jaxili MAF); GPU 1+2 granted; per-
channel RMS whitening frozen from a TRAIN-SAMPLE (not fiducial). 4 arms none/conv/product/both.
WIRED into npe_cnn_nbody_tomo.py: --cnn-map-route flat_local + --cross-op {none,conv,product,both}
(reads autos ch 0-3 only; builds patch-local cross ON-DEVICE in JAX via flatsky_cross.build_channels_jax
roll 0.10; per-channel RMS whiten; same transform for train/val/NDE-compress/obs). Shared helpers
compute_flat_cross_channel_rms + make_flat_cross_transform (importable by build_fiducial_summaries_cnn).
★ GATE A PASS (2026-06-09): A1 (gate_a_flat_cross_cnn.py) all 4 ops PASS — jax-vs-numpy oracle exact
(none/product) / 1.7e-7 FFT-roundoff (conv/both), channels 4/10/10/16, raw autos preserved exact,
whitening exact, deterministic (obs↔train identity). Product channel RMS ~115-167× smaller than autos
(whitening MANDATORY; post-whiten std≈1). A2 throughput (real GPU-1 train, measured): steady-state
none ~93 it/s, product ~75 it/s (batch 128) ⇒ on-device cross costs ~20%, does NOT starve GPU (no
L1-style 251/s collapse). At 75 it/s an 80k-step train ≈ 18 min + 1 one-time compress pass; single
TFDS reader/job. AWAITING Andreas sign-off before launching the 4-arm compressor matrix.

## Loop status (live)
★★ CAMPAIGN RESULT DONE 2026-06-09 (population sweep complete, 9000 obs/arm pooled 3-seed median):
~92% OF THE FULL-SPHERE L1 CROSS GAIN WAS LEAKAGE. flat-local L1 FoM3: auto-only 2405 | +conv 2499
(1.04×) | +product 2875 (1.20×) | +both 2910 (1.21×) vs full-sphere auto+cross 8530 (3.88×). Physical
patch-local cross retains +21%; conv(=alm-product/Zürcher flat-sky analog) +4% ⇒ ~99% of THAT op's
gain was leakage; pointwise PRODUCT (=ξ_ij) survives +20% (σOm/σs8 −9%); both≈product (high-dim NDE
fine in pooled median). Leakage lived in w0 (full-sphere σw0 .246→.188 vs patch-local .245→.232; σOm
physical-both .046 MATCHES full-sphere .046). Auto-only 2405≈full-sphere 2200 validates. POPULATION
MEDIAN OVERTURNS single-obs ranking (single-obs conv+32%>product+17% via per-seed-mean+favorable patch;
pooled median product≈both≫conv). Calibrated TARP✓+SBC✓ (L-C2ST N/A high-dim). Writeup
FLATSKY_CROSS_RESULT.md + memory project_flatsky_cross_deleaked_result. Per-arm medians in
results/exploratory/flatsky_cross_2026_06/population_sweep/<arm>/median_summary.json.
NEXT: CNN arms (jax flat cross + per-ch RMS) → de-leaked L1-vs-CNN. HANDOFF written:
HANDOFF_FLATSKY_CNN_2026-06-09.md (repo root) — start the next session there. L1 side COMPLETE +
calibrated + written up (FLATSKY_CROSS_RESULT.md) + showcase/representative-corner plots delivered.
CNN phase notes (in handoff): wire --cnn-map-route flat_local into npe_cnn_nbody_tomo.py (jax flat
cross on-device, per-channel RMS not frozen-σ, resnet50_gn/plain not BN, example-disjoint perm split,
build-both-slice does NOT transfer so each arm = own compressor; L-C2ST WORKS for CNN 10-dim).

## Loop status (gate-c)
★ GATE C CLOSED 2026-06-09 (Andreas: "fine with just tarp and sbc"): TARP-DRP ✓ + SBC ✓ both pass;
L-C2ST N/A (UNDERPOWERED at high-dim L1: self-test ST_H0 ok median-p 0.98 but ST_H1 +0.5σ-w0 not
detected median-p 0.85 ⇒ logreg can't resolve local miscalib conditioning on 800-3200-dim x with
n_cal=2000; prior campaign ran L-C2ST on CNN 10-dim where it works. Self-test gate correctly aborts
— see memory). ⇒ the de-leaked flat-local cross result is CALIBRATION-TRUSTWORTHY.
CONSOLIDATED HEADLINE (calibrated): patch-local cross gains MODEST — conv +32% FoM3 / σ8 0.89,
product +17% / Ωm 0.77, both +13% (not additive past conv; high-dim NDE), w0 ~unhelped — vs LEAKY
full-sphere auto+cross L1 8530 (~3.9×) ⇒ MOST OF THE FULL-SPHERE CROSS GAIN WAS LEAKAGE. conv→σ8,
product→Ωm operator split (matches GATE B). All single-obs/3-seed; population sweep firms the headline.
NEXT: population sweep (9000 fiducial obs/arm; obs datavectors precomputed fiducial_both_datavectors.npz
→ slice per arm + retrain-in-process sample like TARP → median σ(w0)/2D/FoM3) → full-sphere SUMMARY_PHASE_D
comparison → writeup. THEN CNN arms (jax flat cross + per-ch RMS).

## Loop status (archive)
GATE C TARP-DRP = PASS (2026-06-09, run_flatsky_gate_c_tarp.py → tarp_stratified_val retrain+val-
sample, 4 arms × 3 seeds, FoM3-tercile-stratified). dim-3 (Om,s8,w0): ALL arms + ALL terciles incl
HIGH (tightest cross posteriors) hug diagonal, max|dev| 0.037-0.094 (mostly <0.05), calibrated or
mildly CONSERVATIVE (over-cover), NEVER over-confident ⇒ the modest de-leaked cross gains are REAL,
not over-tight. dim-6: mild dev (<0.10) in weak nuisances h0/ns/Ob (matches prior-campaign known
mild nuisance miscalib); science params clean. Figs gate_c/tarp_drp/figures/tarp_{overlay,per_arm}_dim{3,6}.
Per-pair 2D (mean of seeds, single-obs): conv→σ8 (σ 0.89), product→Ωm (σ 0.77); gains modest (2D FoM
1.1-1.35×); both NOT additive past conv; w0 barely helped. compute_l1_2d_areas.py, plot_l1_matrix_corners.py.
NEXT GATE C: SBC (run_sbc_harm_l1_nobnt.py global rank uniformity) + L-C2ST (lc2st_diagnostic.py local
@fiducial). Then population sweep (9000 obs, datavectors precomputed) + full-sphere comparison + writeup.

## Loop status (history)
L1 MATRIX COMPLETE 2026-06-09 (build-both-slice, all 12 OK; 48min both-build pass cached then 19min
slice+train). FoM3 (pre-GATE-C, mean of seeds 41/42/43):
  auto-only(none) 2354 | conv 3112 (+32%) | product 2759 (+17%) | both 2664 (+13%).
HEADLINE (robust, seed-stable ~5%): patch-local cross gains are MODEST (conv +32%, product +17%)
vs the LEAKY full-sphere auto+cross L1 8530 (~3.9×=+290%) ⇒ MOST OF THE FULL-SPHERE CROSS GAIN WAS
LEAKAGE. The de-leaking test confirms the CROSS_MAP_LEAKAGE_FINDING prediction. conv > product
(conv is the strongest single patch-local cross operator; the smooth large-scale morphology helps).
⚠ both(16ch,3200-dim) 2664 < conv(2000-dim) 3112 — IMPOSSIBLE for info content ⇒ NDE ARTIFACT: the
MAF overfits the high-dim 3200 input (both_s41 early-stopped best@epoch9). both arm UNRELIABLE as-is;
complementarity (does conv+product > each alone?) is CONFOUNDED until the high-dim NDE training is
fixed (more patience/epochs, or L1-VMIM compression — NOT PCA). conv is the clean cross read.
NEXT: (a) investigate/fix both high-dim NDE under-training; (b) σ(w0)+2D per arm (not just FoM3);
(c) GATE C calibration TARP/SBC/L-C2ST BEFORE any contour; (d) population sweep (obs datavectors
precomputed: fiducial_both_datavectors.npz 36000×3200); (e) compare vs auto-only AND full-sphere.

## Loop status (history)
L1 MATRIX = BUILD-BOTH-ONCE-SLICE (run_flatsky_l1_matrix.py, 2026-06-09, 3rd relaunch). The 4 arms
read the SAME autos & 'both'(16ch) is the superset ⇒ none/conv/product are exact column-slices of
'both' (validated bit-identical to ~2e-11 GPU-roundoff). Phase 1: build 'both' datavector ONCE solo
(the expensive loader pass, ~45min, loader-bound ~174/s steady — GATE A2's 486/s was a PREFETCH
ARTIFACT, real steady-state is lower). Phase 2: 11 arm×seed jobs --flatsky-both-cache (slice cols +
slice ranges, obs computed per-op single-map, NO loader pass; NDE-light, pack 4/GPU). Eliminates the
naive version's redundant 4× loader passes + disk-I/O thrash (4 concurrent readers → 40/s).
EARLY PARTIAL (pre-GATE-C, separate-build run, now superseded by single-source): auto-only FoM3
mean ~2323 (≈ full-sphere auto-only 2200 ✓); product mean ~2763 (+19% over auto-only) ⇐ vs leaky
full-sphere auto+cross 8530 (~3.9×) → SUGGESTS most full-sphere cross gain was LEAKAGE. NOT trusted
until GATE C. FoM3 OK to headline now (rule retired 2026-06-09) but report σ(w0)/2D alongside.
NEXT after matrix: pool per-arm FoM3 + σ(w0)/2D, GATE C calibration (TARP/SBC/L-C2ST) BEFORE any
contour, single-obs quick-look (labeled), then population sweep + compare vs auto-only AND full-sphere.
LOADER NOTE: filter decodes 1.13M train→keep 323k (perm 5-6) = 3.5× waste; bumping interleave
cycle_length for FINITE passes is a deferred speedup for the population sweep (don't touch mid-run).

## Loop status (history)
STARTED 2026-06-08 (session 2). Plan signed off by Andreas; 5 design decisions LOCKED:
(1) apodization roll = 10%. (2) L1 cross noise = full per-(channel,scale) σ, FROZEN at fiducial:
add ~32–50 indep noise realizations to the SIGNAL, rebuild cross, wavelet, take per-scale std
across realizations (captures n⊛n AND colored κ⊛n); record in meta; NOT per-patch/cosmo. ALL
channels uniform (autos too). Repo's NOISE-based SNR convention (coeff / propagated-noise-σ), NOT
Zürcher filtered-map-std. (3) product = raw κᵢ·κⱼ, NO ×W²; wavelet handles scales+edges; boundary
handling identical across all channels. (4) slice ch 0–3 of 10-ch TFDS (byte-identical autos to
full-sphere campaign → confound-free; verify ch order + post-preprocess identity in GATE A; autos
are SHT-roundtripped lmax1024). (5) reuse auto n_scales.
INJECTION POINT (verified): wl_stats_torch compute_wavelet_transform → get_noise_levels assumes
WHITE noise per-scale; override stats.noise_levels with frozen per-(ch,scale) σ then recompute
snr_coeffs before compute_wavelet_l1_norms.

PROGRESS (session 2, 2026-06-08):
- GPUs 1+2 both granted this session (Andreas override of GPU-1-only); max them out.
- scripts/sbi/flatsky_cross.py written (np/torch/jax conv+product, bit-identical).
- GATE A1 (operator correctness) ALL PASS: bit-match np/torch/jax conv ~2e-6 / product exact;
  ξ_ij recovery reproduces §14 (diag 0.534→1.039, r 0.479>0.311>0.170); unapod conv mean ~6e-10.
- DATA REALITY (verified): ch0-3 are NOISY SHT-roundtripped autos (no augmentation noise needed);
  perms = independent skies (mean-over-200 → noise floor, ratio 1.05) so freezing needs PAIRED
  realizations; existing channel_empirical_global is total-RMS (signal+noise), NOT noise-based.
- scripts/sbi/freeze_flatsky_cross_noise.py written + RUN. Frozen per-(ch,scale) σ saved to
  results/exploratory/flatsky_cross_2026_06/flatsky_cross_noise_sigma.{npz,json}. R=48, Bessel
  R/(R-1), all 16 channels, faithful sphere-SHT noise (nside512/lmax1024/σ_pix from cache).
- GATE A1b ALL PASS + FINDING: analytic white propagation mis-normalizes even AUTO finest scale
  by ~2× (0.48×, finest starlet ℓ≳1440 > lmax1024 band-limit) and +19% coarsest; conv cross-noise
  per-scale profile departs hugely from white (L1 dist 0.68), product 0.25. ⇒ frozen-empirical
  fixes autos too; auto-only L1 will differ slightly from prior full-sphere campaign normalization.
- scripts/sbi/flatsky_cross_l1.py written (frozen-σ override L1, on-device cross, train/val/obs,
  per-channel SNR-range calibration). GATE A1c ALL PASS: shapes (conv/prod D=2000, both 3200),
  obs↔train bit-match, autos op-independent, conv≠product, SNR not collapsed (median ~1.0 vs old
  bug ~1e-4). FINDING: pointwise PRODUCT is intrinsically heavy-tailed (prod_23 raw max|SNR| 181;
  conv/auto ~25) ⇒ adopted PER-CHANNEL percentile (0.5/99.5) histogram ranges. NOT the old band-aid
  (σ now correct; percentiles only set bin extent, cf. Zürcher fixed [-4,4]). AWAITING Andreas read
  on per-channel-percentile vs fixed-symmetric range before main-script wiring.
- L1 WIRING DONE: --cross-maps-route flat_local + --cross-op {none,conv,product,both} +
  --flatsky-cross-sigma + per-channel frozen percentile SNR ranges (clamp_overflow=True),
  meta provenance block (apod 0.10, frozen σ, ranges, channel source). Smoke run PASSED
  end-to-end (obs L1 2000-dim, calibration, datavector build, route resolution all correct).
- THROUGHPUT FIX: per-channel wavelet loop starved GPU (29% util, 251/s). Batched wavelet over
  channels + channel-chunking (FLATSKY_MAX_WAVELET_MAPS=6144, OOM-proof) → conv 870/s, both 486/s.
- CAMPAIGN MATRIX APPROVED (Andreas): 4 arms (none/conv/product/both) × {L1,CNN} × 3 seeds; L1
  arms FIRST (de-risk); single-obs quick-look (labeled, not a result) before 9000-obs sweep;
  NO contour trusted before GATE C. Scope reminders: final compare vs auto-only AND vs full-sphere
  SUMMARY_PHASE_D (how much gain was leakage); provenance in every meta. (B) fixed-[-4,4] binning
  robustness = BACKLOG.
- ALL CHEAP GATES PASS: A1 operators / A1b noise-σ / A1c L1-module / A2 throughput (cross build
  does NOT starve GPU; rel-to-none = channel-count ratio) / B cosmology-dependence (cross tracks
  σ8 0.96, w0 0.72-0.80, product tracks Ωm 0.78 >> conv 0.43 ⇒ complementary).
NEXT (A2/B checkpoint cleared): build flat_local L1 orchestrator (4 arms × 3 seeds, GPU 1+2,
datavector cached per arm) → launch → GATE C calibration (TARP/SBC/L-C2ST) → single-obs contours
→ population sweep → CNN wiring (jax flat cross + per-ch RMS) → CNN arms. Report at each gate.

## Pointers
FLATSKY_CROSS_BUILD_PLAN.md (steps/gates), FLATSKY_CROSS_REDESIGN_NOTES.md (design+validation S1-14), CROSS_MAP_LEAKAGE_FINDING.md (why), HANDOFF_FLATSKY_CROSS_2026-06-08.md (handoff). Memory: project_cross_map_leakage_fullsphere, feedback_l1_cross_must_use_harmonic_route, feedback_never_pca_l1, feedback_fom3_fragile_use_2d_areas, feedback_gpu1_only, reference_jaxili_checkpoint_reload_truncation.
