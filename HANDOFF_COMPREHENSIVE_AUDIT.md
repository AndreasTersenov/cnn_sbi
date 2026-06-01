# HANDOFF — Comprehensive experiment audit

**Created**: 2026-05-27 by the NDE-swap-test session.

**Purpose**: The next Claude session should perform a thorough, exhaustive audit of every experiment ever run in this project, trace the full configuration of each, identify what we can trust, what was contaminated by bugs, and what scientific conclusions actually hold. The output is a single readable document: `EXPERIMENT_AUDIT.md`.

**Why this is needed**: Over months of work, the project accumulated **~1,200+ posterior files** across ~30 experiment directories, 3 NAS campaign directories, 2 worktrees (one with 426 BNT-era posteriors alone), 44 felt fibers, and 29 documentation files. Configurations drifted silently between runs. Bugs were introduced, fixed, and sometimes reintroduced. Results were interpreted based on FoM3, which we now know is fragile for highly-correlated posteriors. Multiple NDE architectures (sbi_lens RealNVP vs jaxili MAF) were used without always being explicit about which. The current session found several bugs mid-flight (inverted channel normalization, wrong batch sizes) that invalidated runs. We need to know what's real.

**Scope includes ALL eras of the project**: the BNT vs no-BNT parity investigations, the BNT losslessness campaigns, the noise curriculum experiments, the ResNet architecture variants, the L1-norm development (l1norm, l1-jax, l1-cross-maps branches), the VMIM compressor variants, the cross-maps campaigns, the auto-push and auto-cross-push autoresearch campaigns, and the canonical-anchors refresh. Nothing is out of scope.

**Execution mode**: This audit should run autonomously — Andreas does not want to be approving intermediate steps. Read everything, document everything, produce the output document. A felt fiber is not needed; this is a one-shot investigation, not an iterative campaign. If something is ambiguous, flag it in the document rather than asking.

---

## What the audit must cover

### 1. Every experiment directory

Locations to scan (each may contain posterior .npy + .meta.json files):

```
# BNT-era results (the earliest phase of the project)
.worktrees/bnt_tomo_study/                              (426 posteriors — BNT parity investigations)
scripts/sbi/results/final/paper_sbi_consolidation/      (648 posteriors across ~25 subdirs)
  ├── bnt_comparison_tomo4/                             (18 — BNT vs no-BNT direct comparisons)
  ├── cnn_bnt_losslessness_campaign/                    (36 — original BNT losslessness)
  ├── cnn_bnt_losslessness_campaign_cdim{8,10}/         (20 — cdim variants)
  ├── cnn_bnt_losslessness_campaign_indep_split_*/      (40 — the 70/30 split precedent!)
  ├── cnn_bnt_losslessness_campaign_multipatch_*/       (46 — multipatch variants)
  ├── cnn_bnt_noise_curriculum_campaign/                 (82 — noise curriculum experiments)
  ├── cnn_bnt_parity_campaign/                          (56 — BNT parity)
  ├── cnn_bnt_resnet_split_campaign/                    (98 — ResNet architecture + split tests)
  ├── cnn_nobnt_deep_audit/                             (18 — no-BNT deep audit)
  ├── cnn_noiseless_vs_noisy/                           (6 — noise level study)
  ├── nobnt_final_matrix/                               (45 — no-BNT final comparison matrix)
  ├── baryonified_appendix/                             (180 — baryon contamination study)
  ├── smoke_cnn_l1_jaxili/                              (2 — the CNN+jaxili smoke test!)
  └── smoke_l1vmim_jaxili/                              (1)
scripts/sbi/results/final/baryon_bias_tomo4_study/
scripts/sbi/bnt_tomo4_study/                            (study-level JSON manifests)

# L1-vs-CNN comparison era (current focus)
scripts/sbi/results/exploratory/                        (200 posteriors across ~20 subdirs)
  ├── cross_maps_campaign/                              (L1 cross-maps development)
  ├── cross_only_campaign{,_v2_chsigma}/                (cross-only L1/CNN comparisons)
  ├── auto_cross_v2_chsigma/                            (L1 auto+cross with fixed noise model)
  ├── canonical_anchors_refresh/                         (the current campaign)
  ├── apples_v_iter108_autoonly/                        (auto-only apples-to-apples test)
  ├── cnn_with_harm_cross{,_normalized}/                (CNN harmonic cross-maps)
  ├── harmonic_vs_cnn_investigation/
  ├── h1_inductive_bias/                                (attention architecture test)
  ├── h3_cdim_sweep/                                    (cdim=100 test)
  ├── zero_mean_maps_parity_check/                      (mass-sheet degeneracy fix)
  ├── systematic_runs_cnn_retrain_proper/
  ├── systematic_runs_l1_rerun_proper/
  ├── systematic_runs_l1_snr10_rerun/
  ├── probes_configs_comparison/                         (3-probe × 3-config comparison)
  └── cnn_{extended_train_zm,lossiness_check,resnet*,vmim_target_stability}/

# Autoresearch campaigns (NAS-stored)
/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/       (70 posteriors)
/nas/tersenov/claude-notes/runs/cnn-auto-cross-push-18-20-2026/ (30 posteriors)
/nas/tersenov/claude-notes/runs/ldt-l1-discrepancy-2026-05-23/
.worktrees/cnn-auto-cross-push-18-20-2026/                      (141 posteriors)

# Diagnostics
scripts/sbi/results/diagnostics/                        (SBC, TARP, cross-map diagnostics)
scripts/sbi/results/dryruns/
scripts/sbi/results/run_artifacts/                      (pre-campaign legacy checkpoints)
scripts/sbi/results/wandb_runs/                         (16 W&B run dirs from Feb 2026)

# Study-level orchestrators
scripts/sbi/nobnt_tomo_bins_crosscorr_study/
scripts/sbi/optimal_nobnt_crosscorr_benchmark/
```

### 2. For each experiment/run, extract and record

- **Script used** (which .py file, which branch)
- **NDE architecture** (sbi_lens RealNVP or jaxili MAF — this is critical and was often implicit)
- **NDE hyperparameters** (layers, hidden, LR, patience, batch_size)
- **Compressor** (architecture, cdim, checkpoint step, checkpoint policy, what data it was trained on)
- **Data source** (which TFDS variant, harmonic cache or not, field size, npix)
- **Train/val/NDE splits** (train/train overlap? 70/30? disjoint audit?)
- **Preprocessing** (zero-mean-maps, standardize-summary, log1p, PCA, clip)
- **Channel config** (auto-only, auto+cross, cross-only; channel normalization; noise model)
- **BNT** (applied or not, which matrix version)
- **Seeds** (which seeds, how many)
- **Result** (per-seed FoM3, pooled FoM3, 2D areas for Om-s8/Om-w0/s8-w0, marginal σ)
- **Known bugs active at time of run** (PCA default, harmonic route noise model, RealNVP instability, etc.)
- **Whether result is trustworthy** (yes/no/partially, with justification)

### 3. Known bugs and silent configuration issues to check for in every run

1. **PCA default = 50** in L1 script — craters FoM3 by 5×. Must be `--pca-components 0`.
2. **L1 cross TFDS route** silently falls back to broken `auto_scalar` noise model.
3. **RealNVP NDE instability** — diverges after 1-7k steps; val loss curves must be checked.
4. **Train/train compressor-NDE overlap** — inflates FoM3 by some unknown amount.
5. **CNN mass-sheet-degeneracy leak** — pre-`--zero-mean-maps` CNN posteriors overstate precision by ~2× marginals, ~25-30× FoM3.
6. **BatchNorm contamination** on harmonic 10-channel input — stock ResNet BN gives FoM3~700; need GroupNorm.
7. **Compressor checkpoint policy** — `last_step` vs `best_val` can differ significantly.
8. **NDE architecture mismatch** — CNN uses sbi_lens RealNVP, L1 uses jaxili MAF. Different flow families, different capacity, different training dynamics.
9. **Channel normalization** — harmonic_normalize_input_channels (RMS-based for CNN) vs channel_empirical_global (noise-based for L1).
10. **FoM3 fragility** — 1-2% correlation changes → 50% FoM3 changes for highly-correlated posteriors. 2D areas and marginal σ are more stable metrics.

### 4. Documentation to cross-reference

Read these in order of reliability (most reliable first):
- `.meta.json` files (ground truth for what actually ran)
- Log files (`*.log`) in each run directory
- `.felt/` fibers (investigation narratives, may reference stale numbers)
- Root-level `.md` documents (interpretation docs, some cite v1/buggy numbers)
- Memory files in `~/.claude/projects/.../memory/`
- CLAUDE.md (project conventions, some added after bugs were found)

### 5. Git history

Key branches and what they represent:
- `main` — stable baseline
- `l1-cross-maps` — current development branch (L1 vs CNN comparison)
- `bnt-parity-techniques` — earlier BNT campaign
- `autoresearch/cnn-auto-push-18-20-2026` — CNN auto-only push campaign
- `autoresearch/cnn-auto-cross-push-18-20-2026` — CNN auto+cross push campaign
- `bnt_tomo_study` — BNT study branch
- `l1-jax`, `l1-jax-cnn-audit`, `l1-jax-indep-split`, `l1-jax-multipatch`, `l1-jax-resnet` — various L1 development branches
- `l1_compressor`, `l1norm` — earlier L1 work

---

## Key findings from the current session to carry forward

1. **CNN and L1 use different NDE architectures.** CNN: sbi_lens ConditionalRealNVP (8 layers, hidden 256, 567k params). L1: jaxili ConditionalMAF (5 layers, hidden [50,50], 20k-517k params depending on input dim). This was never made explicit in any documentation.

2. **The RealNVP is catastrophically unstable** under the 70/30 canonical split (diverges to NaN at step 2-5k). Under train/train (iter-108), it diverges later (~step 10k) but still diverges. jaxili MAF is stable.

3. **Swapping NDE on the same compressor**: iter-108 compressor + RealNVP = 25.9k FoM3 (single seed s41). Same compressor + jaxili MAF = 17.6k FoM3. But the contours are visually almost identical — the difference is FoM3 amplifying tiny correlation changes.

4. **2D areas and marginal σ are more honest metrics**: the 25.9k vs 17.6k FoM3 gap corresponds to only ~5% wider marginals and ~15-20% larger 2D contour areas.

5. **L1 auto+cross genuinely outperforms CNN auto+cross**, even with the same NDE. Using 2D areas, L1 is 15-36% better on Om-s8, 0-24% on Om-w0, 7-27% on s8-w0. The gap is real but smaller than FoM3 suggested.

6. **CNN cross channels don't help much under jaxili**: auto-only FoM3=16.2k, auto+cross FoM3=17.6k (+9%). Under L1, cross channels give a ~3× boost. The CNN compressor (cdim=10) doesn't extract useful cross-channel information.

7. **The VMIM compressor uses a companion RealNVP** (4 layers, sbi_lens) during training. If this companion NF is suboptimal, the compressor gradients during VMIM training may be suboptimal too. No one has tested VMIM with a MAF companion.

---

## Output format for the audit document

The audit should produce `EXPERIMENT_AUDIT.md` at the repo root with:

1. **Executive summary** (1 page): what we know for sure, what's uncertain, what's invalidated.

2. **Experiment catalog** (the bulk): one section per experiment directory, each with a table of runs, their configs, their results, and trust status.

3. **Bug timeline**: when each bug was introduced, when discovered, which runs it affected.

4. **Clean comparison table**: the subset of runs that can be trusted for L1-vs-CNN comparison, with 2D areas and marginal σ (not just FoM3).

5. **Open questions**: what experiments would actually resolve the L1-vs-CNN question cleanly.

---

## Project chronology (rough phases to organize the audit around)

**Phase 1 — BNT parity (branch `bnt-parity-techniques`, `bnt_tomo_study`)**: The original question was whether the CNN-VMIM compressor preserves information through the BNT transform. Experiments compared BNT vs no-BNT posteriors, tested losslessness across cdim values (6, 8, 10, 12), explored noise curriculum, tested ResNet architectures with different split disciplines. This produced the `paper_sbi_consolidation` results. Key document: `CLAUDE_CODE_HANDOFF.md`, `BNT_NO_BNT_CONTOUR_INFLATION_NOTE.md`. The 70/30 independent split was first used here (`cnn_bnt_losslessness_campaign_indep_split_*`).

**Phase 2 — L1-norm development (branches `l1norm`, `l1_compressor`, `l1-jax*`)**: Adding the wavelet L1-norm statistic as an alternative to CNN compression. Development of the L1 pipeline, jaxili NPE integration, cross-map computation (flat-sky FFT and harmonic full-sphere). Key documents: `L1_CONTOUR_INVESTIGATION_LOG.md`, `L1_FIXES_VALIDATION_REPORT.md`, `L1_VMIM_FINAL_CONCLUSIONS.md`.

**Phase 3 — L1 vs CNN comparison (branch `l1-cross-maps`)**: The current focus. Systematic comparison of L1 vs CNN on auto-only and auto+cross configurations. Discovered the mass-sheet-degeneracy leak (`--zero-mean-maps`), the L1 cross-channel noise model bug, the ResNet BN contamination. Ran the cross-only and auto+cross campaigns, the autoresearch push campaigns, the H1/H2/H3 hypothesis tests. Key documents: `HARMONIC_L1_VS_CNN_INVESTIGATION_*.md`, `CNN_CROSS_MAPS_INFORMATION_NOTE.md`, `HANDOFF.md`.

**Phase 4 — Canonical anchors refresh (current)**: Attempting to get clean, comparable numbers across all four arms. Discovered the train/train contamination in all prior CNN campaigns. The canonical refresh campaign produced new anchors but the CNN auto+cross number dropped 47%, triggering the current audit.

**Phase 5 — NDE-swap test (this session)**: Discovered that CNN uses sbi_lens RealNVP while L1 uses jaxili MAF. Tested swapping the NDE. Found that FoM3 is fragile; 2D areas are more stable. Determined that L1's advantage is real but narrower than FoM3 suggested, concentrated in w₀.

Each phase built on (and sometimes invalidated) results from the previous phase. The audit must trace these dependencies.

## How to run the audit

The audit session should:

1. Start by reading this handoff + CLAUDE.md + MEMORY.md.
2. Write an inventory script that walks all experiment directories and extracts the key fields from every .meta.json into a single CSV/JSON.
3. Group runs by experiment/campaign.
4. For each group, read the relevant logs, fibers, and documentation.
5. Cross-reference with the bug timeline.
6. Write the audit document incrementally, one section at a time.
7. Use the Explore agent for broad searches; read files directly for verification.

Budget: this will take a full long session. Don't rush. Read every meta.json. Check every claim against the actual data.
