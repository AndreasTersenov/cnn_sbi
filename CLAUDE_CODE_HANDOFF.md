# Claude Code Handoff (detailed runbook, 2026-04-21)

This is the detailed restart package for continuing this project in Claude Code with minimal context burn.

---

## 0) One-minute summary

1. **Read first:** `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` (new master synthesis + A&A letter blueprint).
2. **Latest heavy campaign completed:** `cnn_bnt_parity_campaign` (A/B pilots + C confirmations).
3. **Bottom-line result:** paired consistency / domain-adversarial tricks did not generalize reliably to 5-seed confirmation.
4. **Critical code fix already in place:** paired augmentation compatibility in `compress_dataset(...)` (fix for `KeyError: 'maps'`).
5. **Current branch:** `bnt-parity-techniques` (HEAD currently at `4e64a3f` in this environment).

---

## 1) Repository state and safety rules

### 1.1 Branch and dirtiness

- Branch: `bnt-parity-techniques`
- The worktree contains many unrelated dirty/untracked files (notebooks, caches, results, `__pycache__`, etc.).

### 1.2 Golden safety rules for Claude Code

1. **Never use `git add .`**.
2. Stage files explicitly by path.
3. Do not remove or rewrite pre-existing dirty files unless user explicitly asks.
4. Do not run destructive cleanup (`git reset --hard`, mass deletes, etc.).

### 1.3 Files that matter for the latest work

- Code:
  - `scripts/sbi/npe_cnn_nbody_tomo.py`
  - `scripts/sbi/run_cnn_noise_curriculum_campaign.py`
- Narrative/docs:
  - `BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md`
  - `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`
  - `CLAUDE_CODE_HANDOFF.md` (this file)

---

## 2) What was done in the latest cycle

### 2.1 Scientific synthesis for paper writing

Created:

- `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`

This file consolidates:

- L1 diagnosis and fixes,
- L1-VMIM conclusions,
- no-BNT/BNT/baryon consolidated matrix,
- CNN BNT-losslessness retraining,
- multipatch and independent split outcomes,
- noise-curriculum outcomes,
- ResNet campaign tradeoffs,
- parity-techniques outcomes,
- integrated interpretation + A&A letter structure (titles, section flow, figure/table plan, claims/evidence map).

### 2.2 Parity-tech implementation and execution

Implemented features:

- paired no-BNT/BNT consistency training controls,
- optional domain-adversarial invariance head,
- campaign runner support for parity flags,
- per-GPU memory cap map (`--xla-mem-fraction-by-gpu`).

Executed fully:

- Phase A (plain pilot): baseline, consistency, consistency+adv
- Phase B (resnet18 pilot): baseline, consistency, consistency+adv
- Phase C (confirm, seeds 41..45): promoted plain + promoted resnet18 variants

Artifacts root:

- `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_parity_campaign/`

### 2.3 Important runtime bug fixed

- Problem: paired augmentation outputs `{maps_nobnt, maps_bnt}` but `compress_dataset` assumed only `maps`.
- Symptom: training finished, then crash during dataset compression (`KeyError: 'maps'`).
- Fix: `compress_dataset(..., paired_map_view=...)` now selects condition-consistent paired map view (`nobnt` or `bnt`) when paired training is enabled.

---

## 3) High-priority reading order (token-efficient)

### Tier 1 (must-read)

1. `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`
2. `scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md`

### Tier 2 (BNT parity/recovery core)

3. `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`
4. `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/FINAL_NOISE_CURRICULUM_REPORT.md`
5. `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_resnet_split_campaign/CNN_BNT_RESNET_SPLIT_CAMPAIGN_REPORT.md`
6. `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_resnet_split_campaign/resnet_extended_tuning_v2/EXTENDED_RESNET_COMPARISON_REPORT.md`

### Tier 3 (supporting diagnostics)

7. `scripts/sbi/results/final/paper_sbi_consolidation/cnn_noiseless_vs_noisy/CNN_NOISELESS_VS_NOISY_REPORT.md`
8. `L1_CONTOUR_INVESTIGATION_LOG.md`
9. `L1_FIXES_VALIDATION_REPORT.md`
10. `L1_VMIM_FINAL_CONCLUSIONS.md`

---

## 4) Fast metrics snapshot (latest parity campaign)

| Variant | FoM ratio (BNT/noBNT) | inflation | rank |
|---|---:|---:|---:|
| plain_baseline_pilot | 0.8184 | 1.0249 | 0.2065 |
| plain_consistency_pilot | 0.7544 | 1.0585 | 0.3041 |
| plain_consistency_adv_pilot | 0.8350 | 1.0545 | 0.2195 |
| plain_consistency_adv_confirm | 0.7491 | 1.0564 | 0.3073 |
| resnet18_baseline_pilot | 0.0238 | 1.6816 | 1.6578 |
| resnet18_consistency_pilot | 0.6088 | 1.0573 | 0.4485 |
| resnet18_consistency_adv_pilot | 0.2076 | 1.3024 | 1.0948 |
| resnet18_consistency_confirm | 1.4329 | 0.8222 | 0.6106 |

Interpretation:

- pilot improvements existed but were not stable in confirmation.
- no tested invariance recipe delivered robust near-lossless parity across confirmation seeds.

---

## 5) Figure and artifact map for manuscript assembly

### 5.1 Parity campaign overlays (confirmation)

- Plain confirm:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_parity_campaign/phase_c_plain_consistency_adv_confirm/plain_ref/figures/overlay_plain_ref_combined_bnt_vs_nobnt.png`
- ResNet confirm:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_parity_campaign/phase_c_resnet18_consistency_confirm/resnet18_ref/figures/overlay_resnet18_ref_combined_bnt_vs_nobnt.png`

### 5.2 Other key campaign figures/tables

- Losslessness campaign report + summaries:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_losslessness_campaign/CNN_BNT_LOSSLESSNESS_RETRAIN_REPORT.md`
  - `.../comparison_summary.{csv,json}`
  - `.../comparison_refinement_summary.{csv,json}`
- Noise curriculum final:
  - `scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_noise_curriculum_campaign/FINAL_NOISE_CURRICULUM_REPORT.md`
- Final global synthesis:
  - `scripts/sbi/results/final/paper_sbi_consolidation/FINAL_SCIENTIFIC_REPORT.md`

Note: parity campaign outputs are PNGs (no PDFs in that specific tree).

---

## 6) Commands Claude Code can run immediately

### 6.1 Quick integrity checks

```bash
cd /mnt/home/tersenov/software/cnn_sbi
git --no-pager branch --show-current
git --no-pager status --short | head -n 80
```

### 6.2 Recompute parity consolidated CSV (if needed)

```bash
cd /mnt/home/tersenov/software/cnn_sbi
python - <<'PY'
import csv
from pathlib import Path
base = Path('scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_parity_campaign')
runs = [
('phase_a_plain_baseline','plain_baseline_pilot'),
('phase_a_plain_consistency','plain_consistency_pilot'),
('phase_a_plain_consistency_adv','plain_consistency_adv_pilot'),
('phase_b_resnet18_baseline','resnet18_baseline_pilot'),
('phase_b_resnet18_consistency','resnet18_consistency_pilot'),
('phase_b_resnet18_consistency_adv','resnet18_consistency_adv_pilot'),
('phase_c_plain_consistency_adv_confirm','plain_consistency_adv_confirm'),
('phase_c_resnet18_consistency_confirm','resnet18_consistency_confirm'),
]
out = Path('scripts/sbi/results/final/paper_sbi_consolidation/cnn_bnt_parity_campaign/parity_consolidated_summary.csv')
rows = []
for d,label in runs:
    p = base/d/'comparison_summary.csv'
    with p.open() as f:
        r = next(csv.DictReader(f))
    rows.append({
        'label':label,
        'arch':r['compressor_arch'],
        'fom_ratio':r['fom3_ratio_bnt_over_nobnt'],
        'abs_fom_err':abs(float(r['fom3_ratio_bnt_over_nobnt'])-1.0),
        'inflation':r['inflation_std_sum_bnt_over_nobnt'],
        'sigma8_ratio':r['sigma8_std_ratio_bnt_over_nobnt'],
        'rank':r['rank_score'],
    })
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('Wrote', out)
PY
```

### 6.3 Compile check for relevant scripts

```bash
cd /mnt/home/tersenov/software/cnn_sbi
conda run -n jaxili python -m py_compile \
  scripts/sbi/npe_cnn_nbody_tomo.py \
  scripts/sbi/run_cnn_noise_curriculum_campaign.py
```

---

## 7) Commit strategy in a dirty tree

If user asks for commits, use only explicit paths, e.g.:

```bash
cd /mnt/home/tersenov/software/cnn_sbi
git add scripts/sbi/npe_cnn_nbody_tomo.py
git add scripts/sbi/run_cnn_noise_curriculum_campaign.py
git add BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md
git add PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md
git add CLAUDE_CODE_HANDOFF.md
git commit -m "..."
```

Avoid staging generated/cached artifacts unless explicitly requested.

---

## 8) Known pitfalls and context traps

1. **Mixed historical artifacts** exist; do not infer baseline truth from one old posterior file.
2. **BNT difficulty is noise-conditioned**; noiseless comparisons can look misleadingly optimistic.
3. **Parity objectives can conflict** (cosmology sufficiency vs domain invariance), yielding pilot/confirm drift.
4. **Example-level split is not cosmology-level split** in independent-split campaigns.
5. **Worktree is noisy**; always isolate target files.

---

## 9) Suggested next-step tracks

### Track A: Paper-first (recommended if compute pause)

1. Use `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` as base.
2. Draft full A&A Letter text and figure captions.
3. Generate one clean “claims vs evidence” table for supplementary material.
4. Finalize bibliography and methods reproducibility appendix.

### Track B: Method-first (if continuing experiments)

1. Start from parity campaign failure mode.
2. Retune objective weights/schedule with strict pilot+confirm protocol.
3. Require 5-seed confirmation before promoting any new claim.

---

## 10) Copy-paste prompts for Claude Code

### Prompt 1 — manuscript drafting

“Read `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md` and draft a complete A&A Letter manuscript in markdown. Keep every quantitative claim tied to specific artifact file paths.”

### Prompt 2 — clean commits only

“On `bnt-parity-techniques`, prepare minimal commits only for: `scripts/sbi/npe_cnn_nbody_tomo.py`, `scripts/sbi/run_cnn_noise_curriculum_campaign.py`, `BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md`, `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`, `CLAUDE_CODE_HANDOFF.md`. Do not stage unrelated dirty files.”

### Prompt 3 — reproducible parity re-analysis

“Produce `parity_consolidated_summary.csv` from all phase `comparison_summary.csv` files and rank variants by `abs(fom_ratio-1)` then rank score.”

---

## 11) Final single-line checkpoint

If Claude Code only gets one line:

> Start with `PROJECT_SCIENTIFIC_KNOWLEDGE_BASE.md`; parity/invariance work is fully implemented and executed but did not generalize in confirmation, and all evidence is under `scripts/sbi/results/final/paper_sbi_consolidation/`.
