# FoM error-bar recovery on Jean-Zay — state handoff

**Written:** 2026-07-27, after the no-BNT + BNT ℓ1 rows completed and the CNN compressors were launched.
**Supersedes:** `recovery_handoff/HANDOFF_JZ_RETRAIN.md`, which is wrong in several load-bearing
ways (see §8). Trust this file and the surviving artefacts over it.

---

## 1. Deliverable and current status

Error bars for the 8 rows of the L1-vs-CNN table. **6 of 8 complete.**

| Row | Paper | Retrained (ensemble) | Δ | Jackknife bar | Median term | Singles band |
|---|--:|--:|--:|---|---|---|
| ℓ1 auto, no-BNT | 2429 (ens) / 2448 | 2511.1 | +3.4% | **±13.4 (0.52%)** | ±4.14 (0.16%) | ±30 (1.09%) |
| ℓ1 auto, BNT | 388 (ens) | **390.7** | **+0.7%** | **±20.6 (5.10%)** | ±1.00 (0.26%) | ±46 (9.88%) |
| ℓ1 +product, no-BNT | 3009 (ens) / 3045 (single) | 3053.9 | +1.5% | **±91.4 (2.93%)** | ±5.45 (0.18%) | ±185 (5.72%) |
| ℓ1 +product, BNT | 718 | 758.3 | +5.6% | **±17.6 (2.20%)** | ±2.36 (0.31%) | ±37 (4.02%) |
| joint ℓ1, no-BNT | 3371 | ✗ BLOCKED (§7a) | — | — | — | — |
| joint ℓ1, BNT | 2424 (ens) | ✗ BLOCKED (§7a) | — | ±6.74 (0.28%) *from surviving run* | — |
| CNN, no-BNT | 3326 | 3503.5 | +5.3% | **±43.7 (1.24%)** | ±8.20 (0.23%) | ±34 (0.96%) |
| CNN, BNT | 3186 | 3265.7 | +2.5% | **±26.7 (0.81%)** | ±8.06 (0.25%) | ±18 (0.53%) |

All six retrained rows land **+0.7% to +5.6%** of their published values — inside the ~10–15%
agreement the recovery was scoped to. `final_bars.py` prints `MISMATCH` above 2%; that is its own
strict self-check, not a failure of the row.

CNN singles are far tighter than ℓ1 singles (0.53–0.96% vs 1.1–9.9%) — a ResNet-18 VMIM is a more
reproducible compressor than the MLP-on-ℓ1. CNN jackknife bars are correspondingly small.

Results JSON: `recovery/final_bars_all.json`, `final_bars_cnn.json`, `final_bars_surviving.json`,
`final_bars_nobnt.json`, `final_bars_joint.json`.

---

## 1b. WHAT THE ERROR BAR IS — SETTLED 2026-07-28 (read before touching any bar)

Authority: **`NOTE_FOM_ERROR_BARS.md`** in the PRIVATE paper repo
`AndreasTersenov/L1_vs_CNN_Tomographic_SBI` (needs a PAT; mirrored to
`recovery/paper_repo/main/`). It supersedes the older HANDOFF §6 reading.

```
+/-   = mean +/- std over the 3 PRE-ENSEMBLE compressor singles, for ALL rows
        (§5.3, §5.4) -- including rows whose central value is an ensemble.
        Also quote the min-max band. A 3-draw sigma carries ~50% relative
        uncertainty; present as INDICATIVE reproducibility.
bias  = the single->ensemble shift. NOT an error bar (§1, §5.4). Reported
        separately. "the measured de-inflations 779->718, 425->388 ARE the
        bias term, and they are the reason the ensemble is the quoted
        estimator in the BNT rows."
median= block bootstrap over the 180 patches, keeping all 50 reps, 10^4
        replicates, 68% PERCENTILE interval, run once per arm at SEED 41
        (§4, §5.5). Caption term only.
```

**The leave-one-out ensemble jackknife is NOT in the spec.** `final_bars.py` still
computes it; it is a diagnostic and must never be quoted as the bar. Use
`make_table.py` (spec-format output) for anything paper-facing.

Which estimator each published row quotes — from the intact
`RESULT_NOBNT_ENSEMBLE_ROBUSTNESS.md`:

| arm | no-BNT | quoted as | BNT | quoted as |
|---|--:|---|--:|---|
| ℓ1 auto | **2448** (ens 2429) | SINGLE | 388 | ENSEMBLE |
| ℓ1 +product | **3045** (ens 3009) | SINGLE | 718 | ENSEMBLE |
| joint ℓ1 | 3371 | ENSEMBLE | 2424 | ENSEMBLE |
| CNN | 3326 | SINGLE | 3186 | SINGLE (already calibrated) |

The table mixes estimators deliberately: the ensemble is quoted only where the
single failed the calibration battery. **Never compare a retrained single against
a published ensemble, or vice versa** — that error cost a full round of confusion.

Caption wording is fixed by §6 — copy it verbatim, do not paraphrase.

---

## 2. THE ARCHITECTURE (confirmed by the author — get this right)

Every arm's readout is:

    VMIM(compressor seed) -> sbi_lens ConditionalRealNVP 4x128, POOLED over 3 flow seeds

* **ℓ1 rows:** raw ℓ1 datavector -> **MLP compressor (hidden 256,256) trained with VMIM** -> 10-D -> RealNVP.
  Do NOT put a flow on the raw ℓ1 vector: `ESTIMATOR_OPTIMIZATION_RECORD.md` says
  "the same RealNVP craters on the 2000-D L1 vector". Observed here as all-NaN training.
* **CNN rows:** maps -> **ResNet-18 VMIM** -> 10-D -> same RealNVP. Reads the **4 auto maps only**
  (`--cross-op none`, `channel_slice=slice(0,4)`), no cross channels.
* **Single-quoted row** = one compressor, NDE-pooled. Its bar = SPREAD across the 3 compressor seeds
  (report as range, no Gaussian σ from n=3). Quote the audited s41 as the central value.
* **Ensemble-quoted row** = the SAME 3 compressors POOLED (9 flows). Its bar = leave-one-out jackknife.
* So: **3 compressors per (operator, frame)**; their spread is the singles band, their pooling is the ensemble.

---

## 3. Paths

```
ENV      /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili   (python 3.11 venv; freeze: recovery/env_freeze.txt)
CODE     /lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts/   (15 files, SHA256SUMS.txt)
WORKDIR  /lustre/fswork/projects/rech/prk/ulx34io/recovery/
SCRATCH  /lustre/fsn1/projects/rech/nzu/ulx34io/recovery/
  full_sphere_cache_fiducial_10deg/    rebuilt obs cache (200 perms x 180 patches, geometry EXACT)
  sigma_tables/flatsky_cross_noise_sigma.npz   regenerated no-BNT frozen sigma
  l1_datavector_cache/flat_local_{both,product,none}/       raw ℓ1 datavectors, no-BNT
  l1_datavector_cache_bnt/flat_local_{both,product,none}_bnt/   raw ℓ1 datavectors, BNT
  fiducial_both_datavectors_{nobnt,bnt}.npz    obs datavectors (9000, 3200)
  fid_raw_{both,product,none}[_bnt].npz        obs sliced per operator (key "S")
  vmim/{both,product,none}_{nobnt,bnt}_s4{1,2,3}/    VMIM-compressed arms (10-D)
  cnn/{nobnt,bnt}_s4{1,2,3}/                   CNN arms (in progress)
  sweeps/<label>/per_patch_metrics.npz         per-obs metrics + moments
DEAD TREE (source of truth for code, but 30k+ damaged paths)
         /lustre/fsn1/projects/rech/prk/ulx34io/cnn_sbi/
DATA     /lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets/   (TFDS_DATA_DIR)
         /lustre/fsmisc/dataset/CosmoGridV1/   (raw sims, read-only shared)
```

---

## 4. Verified constants (do not re-derive)

```
geometry   n_centers=180  center_nside=64  min_separation_deg=14.2  max_abs_lat=75
           -> recovery/centers_PAPER_180.npy  (confirmed 22/22 by bit-exact reprojection
              vs the cross TFDS, and recorded in scripts/sbi/run_10deg_build.sh line 16)
field      10 deg / 80 px / reso 7.5'   nside 512   lmax 1024
noise      sigma_e 0.26   galaxy_density 10.0   noise_seed_base 12345
           seed formula: base + 100*cosmo_idx + perm   (verified: reprojection matched at 1e-8)
obs        9000 = 180 patches x perms 0..49   (eval uses perm<50; cache holds 200)
FoM3       1/sqrt(det(cov(samples[:, :3])))  param order Omega_m, sigma_8, w_0
           VERIFIED to 1.7e-16 against 8 surviving posterior.fom.json records
preproc    compression stage: log1p-zscore / clip 5 / min-var 1e-5
           downstream sweep:  none / clip 0 / min-var 1e-12
compressor summary_dim 10, hidden 256,256, nf_layers 4, nf_hidden 128, steps 30000, seed 41..43
CNN        resnet18, dim 10, 80k steps, batch 128, lr 5e-4, best_val  (SUMMARY_ARCH.md)
           measured ~70 steps/s => ~19 min train + ~2 min compress per seed
```

---

## 5. How to run a row (working recipe)

```bash
# 1. VMIM compressor (per operator, per frame, per seed)
vmim_from_cache.py --cache-dir <raw cache> --fid-npz <fid_raw_*.npz> \
  --out-cache <arm> --out-fid <arm>/fiducial_summaries.npz \
  --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
  --steps 30000 --seed <41|42|43> \
  --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5

# 2. sweep: one --arm-dir = single row; three = ensemble row
population_sweep.py --arm-dir <arm> [--arm-dir <arm2> --arm-dir <arm3>] \
  --arm-label <label> --out sweeps/<label> \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000

# 3. bars
final_bars.py --row-json rows.json --nboot 10000 --out final_bars.json
final_bars.py --validate      # reproduces surviving numbers exactly; run after any edit
```

---

## 6. What is DESTROYED (all-zero files; not in git; not recoverable here)

| File | Size | Status |
|---|--:|---|
| `train_jaxili_from_compressed.py` | 9672 B | **REWRITTEN** (`rescued_scripts/`), self-check passes |
| `population_sweep_flatsky.py` | 8617 B | **REWRITTEN** as `recovery/population_sweep.py` |
| `final_bars.py` | 0 B | **REWRITTEN**, validated against surviving numbers |
| `flatsky_cross_noise_sigma.npz` (no-BNT) | 5462 B | **REGENERATED**, cross-checked vs intact `_bnt` twin |
| `flatsky_joint_stats.py` | 11710 B | ✗ **BLOCKER** — see §7 |
| `flatsky_joint_stats.cpython-312.pyc` | 16701 B | ✗ also all-zero |
| `build_fiducial_summaries_cnn.py` | 12695 B | ✗ to reconstruct (CNN obs summaries) |
| ALL compressor + NDE checkpoints | — | gone; retraining is the only route |

Note: the 30,488-entry `damaged_paths_for_jz.txt` inventory **undercounts** — several corrupt files
(e.g. the no-BNT sigma table) are absent from it. Verify every input directly.

---

## 7. OPEN ITEMS

### 7a. joint ℓ1 rows — BLOCKED (needs a decision)
I mapped "joint ℓ1" to `--cross-op both` (conv+product). **That is wrong.**
`JOINT_L1_DEFINITION_AND_THEORY.md` defines it as a **2-D pairwise ℓ1-weighted histogram**:
per bin pair (i,j) and scale s, a KxK grid over the joint (u_i,u_j) plane, cells hold
`sum ½(|u_i|+|u_j|)`, over 6 pairs x 5 scales, then VMIM-compressed to 10-D.
Implementation was `flatsky_joint_stats.py` (`pair2d_features(..., weighted=True)`, `stat="jointl1"`)
— destroyed, source and bytecode.

Evidence it is wrong: my `both` ensemble gave BNT 703.9 vs paper 2424 (3.4x low), while the
surviving `bnt_campaign/l1_both_s41` single-obs FoM3 is 751 — i.e. I faithfully reproduced the
`both` arm; it simply is not the table's row. The no-BNT `both` result (3130 vs 3371) looked fine
and HID the error; only BNT exposed it.

**ASK THE USER:** does `flatsky_joint_stats.py` exist off-cluster? If not, reconstruct from the
definition doc + intact `build_flatsky_joint_arm.py`, and validate against surviving `jointl1_*`
arm metadata before trusting any number. Risky: bin edges, percentile ranges, the ½ weighting,
pair ordering are all places a silent difference hides.
Discard: `sweeps/joint_l1_nobnt_*`, `sweeps/joint_bnt_*`, `vmim/both_*`.

**Surviving evidence found 2026-07-28 (use it as the reconstruction target):**
* `analytical_nde_match/jointl1_bnt_ensemble/per_patch_metrics.npz` is INTACT, N=9000,
  median FoM3 **2424.3** = the published 2424. Its **median term is therefore already
  measurable without rebuilding anything: ±6.74 (0.28%)** — see `final_bars_surviving.json`.
  Only the jackknife term needs the retrain (that file predates arm_mean/arm_cov).
* `calib_sweep_jointl1/SWEEP_RESULT.md` records the readout as **RealNVP 4×128**, and a
  "baseline jointl1 4×128 / 3-seed = FoM3 3754" against "l1+product 3045, CNN 3326".
* **`pair2d_*` is NOT the same statistic** — `pair2d_rnvp_s4{1,2,3}` give 4922/5156/4513
  (N=1000). Do not treat `pair2d_features(weighted=False)` as the jointl1 row.
  `gate_pair2d_rnvp/verdict.json` survives and may pin the distinction.

### 7b. auto no-BNT singles run high — LARGELY RESOLVED (2026-07-28)
Original worry: singles 2777/2740/2800 vs paper 2448 = +13.4%. **That was the wrong comparison.**

Surviving published runs (`analytical_nde_match/*/per_patch_metrics.npz`, all N=9000) read back as:

| surviving run | median FoM3 | published |
|---|--:|--:|
| `auto_nobnt_ensemble` | **2428.6** | 2429 |
| `auto_bnt_ensemble` | **388.4** | 388 |
| `jointl1_bnt_ensemble` | **2424.3** | 2424 |
| `l1product_rnvp_s41_n9000` | **3044.9** | 3045 |

So the table's numbers are reproduced exactly by surviving artefacts — but they are *not all the
same kind of quantity*: 388 / 2424 are ENSEMBLES, 3045 is a SINGLE (s41, N=9000).
Nothing surviving equals 2448; the nearest is the ensemble 2428.6 (**+0.8%**).

And across every retrained row the ensemble sits BELOW the singles (pooling 9 flows widens the
posterior): auto no-BNT 2511 vs singles ~2777; +product no-BNT 3054 vs ~3232; +product BNT 758
vs ~912. So a single running ~11% above its ensemble is the expected pattern, not an anomaly.
Reading 2448 as an ensemble-like number, the retrained 2511.1 is **+2.6%** — in line with the
other rows. Residual question for the author: which run produced exactly 2448.

**Do not** compare a retrained singles median against a published ensemble number again.

### 7c. CNN rows — in progress (2026-07-28)
`build_fiducial_summaries_cnn.py` is **RECONSTRUCTED** at `recovery/build_fiducial_summaries_cnn.py`
and its **G1 gate passes**: max|Δ| = **6.3e-05** (no-BNT) / **1.6e-04** (BNT) vs tol 9e-4.

Why it is exact rather than guessed — the training run persists everything the obs path needs:
* `cache/cnn_obs.npz['x']` = the driver's own compressed observed vector (perm 0, patch 0).
  **That IS the G1 reference** — no log scraping.
* `cache/cnn_cache_meta.npz['info_channel_scale']` = the frozen per-channel RMS. The driver
  added this key precisely because it "previously lived only in stdout logs", so
  `compute_flat_cross_channel_rms` never has to be re-run.
* the same npz carries op / roll_frac / flatsky_bnt / arch / dim / head_width / v2 / checkpoint
  path + **sha256**. The script re-hashes the checkpoint and refuses to run on a mismatch, so an
  arm can never be evaluated with another seed's weights.

The obs path is `load_observed_from_harmonic_cache(..., channel_scale=None,
channel_slice=slice(0,nbins)) -> make_flat_cross_transform(op, RMS, roll, bnt) ->
compressor_eval.apply`. Driver functions are imported, not reimplemented. Build = 20 s for 9000.

Note `harmonic_regime` is **`nobnt` even for the BNT arm** — flat_local reads RAW autos from the
no-BNT cache and applies BNT on-device (order: noise → demean [both in cache] → BNT → cross-build
→ whiten). Confirms CNN and ℓ1 read the same obs cache.

**Preproc differs from the ℓ1 rows** — see §9. Measured s41 singles:
no-BNT 3539.0 (`zscore`) / 3427.6 (`none` control) vs paper 3326; BNT 3384.8 (`zscore`) vs 3186.
Using `zscore 5.0`; the control shows the choice moves the answer by only 3.2%.

---

## 8. Where the ORIGINAL handoff is wrong (cost real time)

1. **§2 TFDS**: the paper used `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`
   (10°/80px, on disk). NOT `grid_20deg_160px_nonoverlap48`. No rebuild was ever needed.
2. **§5 recipes**: taken from the May 20° `launch_canonical_all.sh`, a different campaign.
   Wrong SNR ranges, batch size, route, and architecture.
3. **§5 CNN arch**: `plain` is wrong; it is **resnet18** (plain = 3139, resnet18 = 3326).
4. **§1 "code survived in git"**: 10 modules were never in git; 2 more are stale there.
5. **§3 `requirements.txt`**: does not exist.
6. **§6 "only fom3 + marginal σ saved"**: `fom2d` was saved too.
7. **The "5.8% seed term" is not a valid measurement**: it mixes s41 at N=9000 with s42/s43 at
   N=1000. Self-consistent value from surviving data is **3.87%**. Retrained rows now give
   1.1–9.9% depending on row.

---

## 9. Gotchas that cost time (do not rediscover)

* **numpy tie-order**: `_build_non_overlapping_centers` uses `np.argsort` (quicksort, unstable) and
  every HEALPix ring is a tie block — numpy 1.24.4 and 1.26.4 give tilings sharing only 2 of 180
  centres. Always pass `--centers-npy centers_PAPER_180.npy`; never regenerate.
* **`wl_stats_torch` PyPI v0.1.0 is stale** — lacks `subtract_coarse_mean`. Install from
  `git+https://github.com/AndreasTersenov/wl_stats_torch.git` (env pins commit 77eafc84).
* **`sbi_lens` needs a 2-line patch**: `tfp.experimental.substrates.jax` no longer exists; use
  `from tensorflow_probability.substrates import jax as tfp`. The surviving Learn2Map venv had
  exactly this patch — copy it, don't invent it.
* **`tensorflow-metadata` must be 1.16.1 + protobuf 5.29.3**; newer breaks tfds under TF 2.18.
* **prk `$WORK` is at 91.6% of its 500k inode cap** — put envs/caches on nzu `$WORK` or `$SCRATCH`.
* **`qos_gpu_a100-dev`**: 2 h wall, **max 10 submitted jobs per user**. Everything so far fits in
  2 h (longest: 29 min). Do not pad to 20 h on `t3` — that queues behind ~2400 jobs.
* **Login node**: 5 GB / 1800 s CPU. Never `np.load` the 8–13 GB caches there; it gets OOM-killed.
* **`--fiducial-obs-cache-dir`** (ℓ1 driver) wants the cache ROOT; the loader appends `regime/obs`.
  The CNN driver's equivalent flag is **`--fiducial-obs-cache`** (no `-dir`).
* **The ℓ1 driver's `--fiducial-summaries-out` is not flat_local-aware** (bins with scalar SNR
  ranges while obs uses per-channel) — its own G1 gate rejects it. Use the precompute script.
* **CNN and ℓ1 rows need DIFFERENT sweep preproc.** The CNN driver z-scores its compressed
  summaries before the flow (`--standardize-summary` defaults ON, `--summary-clip-value 5.0`);
  the ℓ1 driver has no such flag at all, so its compressed summaries go in raw. Hence
  ℓ1 sweeps use `--preproc-transform none --clip-value 0`, CNN sweeps use `zscore 5.0`.
  (Measured sensitivity: 3539 vs 3428 on cnn_nobnt_s41 — 3.2%, so this is a fidelity choice,
  not a make-or-break one.)
* **`population_sweep.py` auto-detects the cache prefix** (`l1_*` from `vmim_from_cache.py`,
  `cnn_*` from the CNN driver) via `arm_paths`; pass the arm dir either way.
* **`sw_cnn.slurm` writes `logs/sw_cnn_<jobid>.out`** — `--job-name` does not rename the log,
  so grep by job id, not by name.

---

## 10. Monitoring

A persistent `Monitor` watches SLURM terminal states. If restarting a session, re-arm with a poll
loop over `squeue`/`sacct` that emits on **every** terminal state (COMPLETED/FAILED/CANCELLED/
TIMEOUT), not just success — silence must not be mistaken for progress.
