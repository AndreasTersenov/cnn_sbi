# ℓ1+conv and ℓ1+conv+product — Table 1 error bars, added 2026-08

Adds two arms to the Table-1 error-bar sweep, filling in the two rungs of the operator
ladder (Fig. showing 2448 → 2671 → 3045 → 3255 → 3371) that had no error bars and no
BNT counterparts. Follows the same recipe as every other ℓ1 arm — only the input cache
differs.

Key result: the ℓ1+conv+product no-BNT run uses **the same sweep directories**
(`sweeps/joint_l1_nobnt_*`, `sweeps/joint_bnt_*`, `vmim/both_*`) that were initially
mislabelled as joint ℓ1 during the first recovery. The statistic there was always
correct — `--cross-op both` really is `autos + conv + product` — only the *label* was
wrong. Those runs are the ℓ1+conv+product data and are used as such.  Earlier README
sections marking them "superseded / DO NOT USE" refer to that mislabelling and are
outdated; the data is the real ℓ1+conv+product arm.

| step | job | notes |
|---|---|---|
| slice conv | `slice_conv.slurm` (CPU, ~5 min) | Column-slices ℓ1+conv (10 ch) out of the existing 16-ch `both` cache in both frames. Auto block verified byte-identical to the `none` arm. |
| VMIM + sweep | `conv_arm.slurm <nobnt\|bnt>` (~70 min each) | 3 compressor seeds + 3 singles + ensemble, chained. |
| bars & marginals | `make_table.py --rows rows_table1_v2.json` | 12-row table; see `TABLE1_ERRORBARS_V2.{md,json}` and `TABLE1_CONV_ARMS.{md,json}`. |

**Validation.** conv no-BNT seed 41 reproduces the published `operator_realnvp/conv/n9000`
= 2671 to +1.8%, the closest agreement of any ℓ1 arm in the recovery — an independent
check that the column-slicing approach was sound.

**BNT rows are new.** The published operator ladder (`operator_realnvp/`) is no-BNT
only. `conv` and `conv+product` BNT centrals therefore come from the retrained pipeline
with no published counterpart; flag those two rows as new in the paper caption.

## NDE-variance investigation (jobs `nde_variance*.slurm`)

Not part of the Table-1 sweep — a separate check that the ± convention actually
captures pipeline reproducibility. Re-sweeps a fixed compressor with a different set
of pooled flow seeds (51/52/53 or 61/62/63 instead of 41/42/43). Finding: for the CNN
the flow-set term (0.74% no-BNT, 1.42% BNT) is 2-3× the compressor term, so the
quoted CNN bar under-represents reproducibility. For the ℓ1 arms the flow-set term is
0.3-1.5% and never dominates the compressor term. Per-row shifts are documented in
`RESULT_FOM_ERRORBARS.md`.
