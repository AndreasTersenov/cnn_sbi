# Overnight menu 2 — derived results (PLAN_OVERNIGHT_MENU_2.md)

Screening = 1 seed/3000 obs; full = 3 seeds/9000 obs (always-escalate by design).
Baselines: l1-auto 2405 | l1+product 2875 | l1-BNT 364 | pair2dq_nobnt 2794 (ratio 0.522).

| arm | screening FoM3 | full FoM3 | σ(Om) | σ(s8) | σ(w0) |
|---|---|---|---|---|---|
| A1_pair2d_vmim | 3910 | 3822 | 0.042 | 0.061 | 0.216 |
| A2_pair2d_k8 | 2943 | 2874 | 0.046 | 0.075 | 0.231 |
| B0_bntcut_l1 | 271 | 268 | 0.095 | 0.200 | 0.342 |
| B1_bntcut_sums | 614 | 596 | 0.079 | 0.150 | 0.313 |
| B2_bntcut_deep2 | 624 | 613 | 0.080 | 0.147 | 0.318 |
| B3_nobnt_unicut | 336 | 337 | 0.092 | 0.191 | 0.347 |
| C1_pair2d_bnt_ar | 1534 | 1485 | 0.057 | 0.100 | 0.248 |
| C2_pair2d_k15 | — | 2455 | 0.045 | 0.071 | 0.226 |
| C3_pair2d_k15_bnt_ar | — | 1343 | 0.056 | 0.095 | 0.242 |
| D1_l1_product3 | FAIL | — | — | — | — |

## Branch-sentence resolution (bands registered in the plan BEFORE data)

- **A1** (VMIM): FoM3 3822, TARP devs {'HIGH': 0.08497500000000047, 'MID': 0.08274999999999982, 'LOW': 0.05012499999999892} -> PATHOLOGY-NOT-DIMENSIONALITY: still miscalibrated at 10-d; tracks the statistic/posterior geometry itself.
- **A2** (K=8): TARP devs {'HIGH': -0.06999999999999845, 'MID': -0.049649999999999694, 'LOW': 0.07599999999999979}, FoM3 2874 -> SPARSITY-DRIVEN: worst dev shrinks >=30% vs K=10's -0.134; coarser grids are the calibratable regime.
- **K-trend** [FoM3, worst HIGH dev]: K=8 [2874, -0.070] | K=10 [2794, -0.134] | K=15 [2455, -0.069]
- **B2/B3 = 1.82** (B2 613 / B3 337) -> BNT + two cleaned recombinations costs <=10% of the information while retaining per-slice systematics rejection — the constructive resolution of the BNT trade-off.
- B0/B3 = 0.79 (268)
- B1/B3 = 1.77 (596)
- B3 vs uncut l1-auto: 0.14 (what the uniform cut costs a noBNT analysis)
- (schedule-conditional numbers: M = '3,4;2,3,4;1,2,3,4;0,1,2,3,4', U = '3,4;3,4;3,4;3,4')
- **C1** BNT-adaptive pairwise ratio r = 0.532 (fixed-grid 0.522; registered band 0.52 < r < ~0.75) -> pairwise statistics are structurally more basis-fragile: marginal incompleteness, not placement, dominates.
- **C2** K=15 noBNT: 2455 (-12.1% vs K=10) -> finer binning gives LOWER FoM3 — the curse-of-dimensionality / NDE-estimation artifact (more cells = sparser = harder for the MAF), NOT a lower physical ceiling (a K=15 grid refines K=10, so by the data-processing inequality it contains >= the information). The FoM3 ladder here tracks estimator difficulty, not information content.
- **C3** K=15 BNT-adaptive ratio r = 0.547 (vs K=10 adaptive 0.532; staircase test, threshold +0.05) -> the shear residual is K-stubborn; only a learned linear front-end transports.

Gate dumps/curves: overnight_menu_2/gate_c/. Logs: overnight_menu_2/logs/.
Packing: packing_benchmarks/PACKING_RESULT.md.
A3 (pooled TARP) was resolved pre-launch: see GATE_C_JOINT.md Addendum 2.
