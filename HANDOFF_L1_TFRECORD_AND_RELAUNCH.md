# HANDOFF — Add TFRecord support to the L1 cross script, then relaunch L1 auto+cross

**Created**: 2026-05-28 by the session that ran the L1 auto-only arms + killed the broken L1 auto+cross phase.
**For**: a fresh session continuing the "definitive L1 vs CNN comparison" campaign.
**Read this entire doc before doing anything.** It is exact on purpose. Do not guess; verify each claim against the live files.

---

## 0. TL;DR — your primary task

The L1 auto+cross arms are bottlenecked by GIL-bound `.npz` decompression in the L1 cross script's harmonic-cache loader (~2.4–8 it/s, GPU starving). A validated TFRecord copy of the harmonic cache already exists (`/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid`, 9184 shards, bit-exact, NONE compression) and the **CNN path that uses it is complete and documented** in `scripts/sbi/HARMONIC_TFRECORD_README.md` (~7.4× faster). **The L1 cross script does not read it yet** — that's your job. Reuse the CNN parse logic; do NOT copy its global-shuffle/DLPack (§1, §1a).

**Your job (option B, chosen by Andreas):**
1. Add TFRecord reading to the L1 cross script's `iter_harmonic_examples` (so all its callers speed up), preserving the L1 data semantics **exactly** (bit-identical L1 datavectors).
2. Validate with a bit-exact L1-datavector equivalence test (`.npz` vs TFRecord). This is the gate.
3. Relaunch the L1 auto+cross arms (arm 1 full-train + arm 2 70/30) on TFRecord, 3 seeds × 3 perms each = 18 posteriors.
4. Compute FoM3 + marginal σ + corner overlays; send Andreas the plots.

Do this **properly and consistently** — Andreas's explicit standard. No shortcuts. The bit-exact test is non-negotiable: any numerical drift corrupts the very comparison this campaign exists to make clean.

---

## 0a. ⚠️ WORK SPLIT (decided 2026-05-28) — who does what

This work is split across two sessions:

- **PART 1 — "L1 TFRecord adaptation" → the TFRecord/DLPack session** (it's warm, has the format context). Scope: §1a + §3 + §4 + **§4b (datavector-reuse, the dominant speedup)** + §5 (validation gate). I.e. *make the L1 cross script read TFRecord AND reuse the datavector*, fully validated bit-exact. Routed via `BRIEF_L1_TFRECORD_ADAPTATION.md`, which points back into these sections.
- **PART 2 — "run continuation" → a separate session (this handoff's original purpose)**. Scope: §6 (relaunch L1 auto+cross on the fast path), §7 (CNN-side continuation: compressor regime decision, Phase 0a/0b, CNN NDE arms), Phase C analysis. **Precondition: Part 1 landed + validated.** Part 2 verifies Part 1 (the L1 datavector equivalence test passes, `--harmonic-tfrecord-dir` works) before launching.

If you are the **Part 2** session and Part 1 is not yet done, STOP and check with Andreas — do not run L1 auto+cross on the slow `.npz` path (that's what we just killed).

---

## 1. The CNN TFRecord path is DONE — it's your reference (read its README first)

The parallel session **finished** the CNN-side TFRecord + DLPack work. **Read `scripts/sbi/HARMONIC_TFRECORD_README.md` before coding** — it's the authoritative runbook for the completed CNN path. Summary of the final state:

- **CNN reader** `build_harmonic_tfrecord_iterator` (in `scripts/sbi/npe_cnn_nbody_tomo.py`, ~line 1345): yields `maps` as a **JAX device array via DLPack** (zero-copy, on GPU), `theta` as numpy with H0/100 applied on host, **in-graph LR/UD flip**, **global shuffle (buffer 4096)**. Hits **~17 it/s (7.4× over `.npz`)**; 80k-step compressor ~9.3 h → ~1.3 h.
- **Shared NaN guard** is now `_array_has_nan` (`jnp.isnan` for device arrays, `np.isnan` for numpy) — the `.npz`/TFDS/paired paths are byte-for-byte unchanged. **This is already committed; do not re-edit it.**
- Full `nobnt` regime built at `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid`: **9184 shards** (train 6293 / val 2800 / obs 91), **421 GB, NONE compression**, one shard per `.npz` (identical stem, 48 patches each), manifest `tfrecord_manifest.json` with a content hash.
- CNN usage: `--full-sphere-cross-cache <.npz>` is **still required** (channel-RMS norm, observed datapoint, split-overlap audit); `--harmonic-tfrecord-dir <tfrecord>` only redirects the **compressor-training iterator**. Compression auto-read from the manifest. Recorded in `.meta.json` as `harmonic_tfrecord_dir`, deliberately **not** in the cache fingerprint.

**Rules for your L1 port (it lives ONLY in `scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py`):**
- **Reuse the CNN reader's stable parse logic** (`tf.io.decode_raw` + reshape; the Example feature schema). Do **NOT** copy the DLPack / device-array / in-graph-flip / global-shuffle parts — those are CNN-VMIM-specific (see §1a). The L1 reader yields **NUMPY** (the L1 wavelet stats run in **PyTorch** via `torch.from_numpy`).
- **Do NOT edit the shared NaN guard or any shared training code.** The L1 NDE is `jaxili` with its own handling. Your change is additive to the L1 file only.
- **Run `git status` first** to confirm the CNN work is committed/clean before you start, so you're reading the final reference.

### 1a. ⚠️ The global-shuffle caveat — and why the L1 port must NOT copy it

The CNN reader deliberately uses a **global shuffle** (4096 buffer → ~8 cosmologies per 128-batch vs the `.npz` pool's ~3.7). **VMIM compressor training is batch-composition-sensitive**, so the TFRecord-trained CNN compressor has **different (better) training dynamics** than the `.npz`-trained one. Per the README, Andreas **approved** this and the rule: *"to compare fairly, re-run BOTH the L1 and CNN arms under the TFRecord path."*

Two consequences you must honor:

1. **For the L1 port: preserve sorted shard order — do NOT introduce a global shuffle.** L1 has **no VMIM compressor**; the L1 datavector is a deterministic per-patch transform, and the `jaxili` NDE shuffles the dataset internally with its own seed. So the L1 TFRecord path should be a **speed-only, results-neutral** change: yield shards in the **same sorted-stem order** as the `.npz` reader so the datavector dataset is row-for-row identical and the NDE result is unchanged. The global-shuffle dynamics issue simply does not apply to L1 — and copying it would needlessly perturb the L1 result. (Hard gate is still the bit-identical datavector SET, §5.)

2. **For the CNN compressors already trained this session (campaign-level flag, see §7):** the two RealNVP-companion compressors trained earlier were trained on **`.npz`** (pool shuffle, ~3.7 cosmos/batch). They are therefore **NOT dynamically comparable** to any TFRecord-trained (global-shuffle) compressor. Per Andreas's approved rule, the CNN arms should be **re-run under the TFRecord path** for the final comparison. This means the `.npz`-trained auto-only + auto+cross RealNVP compressors likely need **retraining on TFRecord** (fast now) before the CNN NDE arms are run — so all CNN compressors share one regime. This is CNN-side work (§7), not your L1 task, but the handoff flags it so it isn't missed.

---

## 2. Current campaign state (verify against reality before trusting)

**Campaign**: definitive L1 vs CNN comparison. 10 arms (3 L1 + 5 CNN-RealNVP + 2 CNN-MAF), jaxili MAF NDE for all, 3 seeds × 3 perms.
- **Fiber (constitution)**: `.felt/definitive-l1-vs-cnn-2026-05/definitive-l1-vs-cnn-2026-05.md` — read its "Loop Status (live)" first.
- **Implementation plan**: `~/.claude/plans/mighty-tumbling-sparrow.md`.
- **Audit (background)**: `EXPERIMENT_AUDIT.md` (repo root) — the confounds this campaign eliminates.
- **TFRecord spec**: `scripts/sbi/HARMONIC_TFRECORD_IMPLEMENTATION_SPEC.md`.
- **TFRecord README (completed CNN path — your parse reference)**: `scripts/sbi/HARMONIC_TFRECORD_README.md`.
- **Memory** capturing the caveat: `memory/project_harmonic_tfrecord_training_path.md` (the global-shuffle / not-a-drop-in note).
- **Output root**: `scripts/sbi/results/exploratory/definitive_comparison/`.

**DONE (trust these — artifacts on disk):**
- Both RealNVP-companion CNN compressors trained (80k steps, best-val) + diagnosed:
  - auto-only: `compressors/auto_rnvp/...`; compressed cache `compressed/auto_rnvp_split70/` (cnn_train/val/obs.npz). best-val step 58k, val drift +0.15 nats, max feature corr 0.96.
  - auto+cross: `compressors/autocross_rnvp/.../harmonic_nobnt_ch10/`; compressed cache `compressed/autocross_rnvp_split70/`. best-val step 54k, val drift +0.57 nats, max feature corr 0.86.
  - Diagnostics: `compressors/{auto_rnvp,autocross_rnvp}/diagnostics/` (reusable script: `scripts/sbi/diagnose_compressor.py`).
- **L1 auto-only, both splits, 3 seeds each (DONE):**
  - `posteriors/l1_auto_fulltrain/` (s41,42,43): pooled FoM3 **10,452**; per-seed 12,028 ± 2,387; σ(Om,s8,w0)=0.0355/0.0478/0.1740.
  - `posteriors/l1_auto_split70/` (s41,42,43): pooled FoM3 **8,086**; per-seed 8,997 ± 1,952. **Split penalty ~24%.** (This `l1_auto_split70` arm was an addition beyond the original 10; it completes the {auto,auto+cross}×{full,70/30} L1 matrix.)
  - Corner overlays + FoM3 dashboard: `figures/early/`.
- **TFRecord of the harmonic cache (VALIDATED):** `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid/nobnt/{train,val,obs}/` — 6293/2800/91 shards, compression **NONE**, manifest `tfrecord_manifest.json`. All 5 tests pass (equivalence bit-exact max-abs-diff 0.0; split slicing == .npz; 3.34× throughput). Build script: `scripts/sbi/build_harmonic_tfrecord.py`. Tests: `scripts/sbi/tests/test_tfrecord_*.py`.
- **L1 split-slicing bug FIXED** in `npe_l1norm_cross_jaxili_nbody_tomo.py`: `_list_harmonic_cache_files` now parses `train[70%:]` (added `_parse_harmonic_split_slice` + regex, mirroring the CNN script). Verified: `train[70%:]` → 1888/6293 files, matching the CNN NDE split. (This is already in the file — do not redo it.)

**KILLED:**
- The overnight launch script `launch_phase_a_overnight.sh` (PID was 58287) — killed because its L1 auto+cross phase had (a) a broken arm 2 (the split bug, now fixed) and (b) a slow arm 1 (the `.npz` bottleneck). **Do NOT rerun that script** — it also contains a redundant arm-3 rerun. You will write fresh relaunch commands (§5).

**BLOCKED (later work, NOT your primary task):**
- CNN NDE arms (4–10): need `scripts/sbi/train_jaxili_from_compressed.py` (Phase 0a) — **not created yet**.
- MAF-companion compressors (arms 7,8,10): need `--vmim-companion-backend maf` in the CNN script (Phase 0b) — **not created yet**.
- See §7 for these. Do them only after the L1 task is complete and validated, or hand to another session.

**GPU/CPU policy (UPDATED 2026-05-28 by Andreas):** **GPU 1 — max it out** (sole-tenant; `--xla-mem-fraction`/`XLA_PYTHON_CLIENT_MEM_FRACTION` up to ~0.95). **GPU 0 — use up to 45% only** (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`; it has another tenant now). GPUs 2 and 3 are other users' — do not touch. Up to 50 CPUs. Always `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Check `nvidia-smi` before launching; pin with `--cuda-visible-devices`. Practical split for the two L1 auto+cross arms: put the heavier load on GPU 1, the lighter on GPU 0 (≤45%).

---

## 3. The L1 cross harmonic-cache read sites (what to change)

All L1 harmonic reads funnel through one function, so you change it once:

| Site | Line (approx) | Role | Reads |
|------|------|------|-------|
| `iter_harmonic_examples` | 401 | **THE per-realization reader** — yields `(maps (n_patches,H,W,C) float32, theta (6,) float64, path)` per cache file | the .npz files |
| `_load_harmonic_file` | 377 | pure-numpy load + channel_slice + channel_scale (helper for above) | one .npz |
| `_list_harmonic_cache_files` | 325 | sorted+sliced file list (split-slicing already FIXED) | dir listing |
| `compute_l1_dataset_from_harmonic_cache` | 698 | **the bottleneck**: walks ALL train/val realizations → L1 datavectors | calls `iter_harmonic_examples` (line 767) |
| `calibrate_channel_noise_sigma_from_harmonic_cache` | 537 | per-channel σ from ~32 realizations | calls `iter_harmonic_examples` (line 558) |
| `calibrate_snr_range_from_harmonic_cache` | 583 | SNR range from ~N realizations | calls `iter_harmonic_examples` (line 624) |
| `load_observed_from_harmonic_cache` | 469 | one obs file | direct `np.load` |

**Strategy (cleanest, minimal surface):** add an optional `tfrecord_dir: Path | None = None` parameter to `iter_harmonic_examples`. When set, it yields per-**shard** `(maps, theta, path)` from TFRecord instead of `.npz`, with **identical** post-processing. Because all three callers (dataset + both calibrations) go through `iter_harmonic_examples`, they all get the speedup and stay mutually consistent automatically. Thread a new `--harmonic-tfrecord-dir` (+ `--harmonic-tfrecord-compression`) CLI flag through to these callers.

`load_observed_from_harmonic_cache` (one obs file) can **stay on `.npz`** — it's a single fast read and the `.npz` and TFRecord are bit-identical, so the observed datavector is unchanged either way. (Porting it too is optional and low-value; if you do, keep it bit-exact.)

---

## 4. The exact contract `iter_harmonic_examples` must preserve (DO NOT DEVIATE)

The TFRecord branch must produce, per yielded item, **exactly** what the `.npz` branch produces. Differences here silently corrupt the comparison.

1. **Yield granularity + ORDER**: one yield per **shard** = the 48 patches of one realization, as `maps (48, 160, 160, C)` float32, in **sorted-stem order** (same as the `.npz` reader). NOT shuffled cross-file batches, and **NO global shuffle** (the CNN reader uses a global shuffle deliberately — DO NOT copy that; see §1a for why it's wrong for L1). The L1 consumer batches realizations itself downstream via `realizations_per_batch`, and `jaxili` shuffles the NDE dataset internally — so sorted-order shards keep the L1 result a results-neutral, speed-only change.
2. **Output type**: **NUMPY** float32 for `maps`, **NUMPY** float64 for `theta`. (PyTorch consumes via `torch.from_numpy`. Not JAX device arrays.)
3. **theta**: the **RAW** theta from the record, shape `(6,)`, `[Om, s8, w0, H0, ns, Ob]` with **H0 NOT divided by 100** (e.g. 68.5). The L1 pipeline divides H0 elsewhere — the `.npz` reader yields raw theta (see its docstring: "h_0 not yet divided by 100"). Match that. ⚠️ This is the OPPOSITE of the CNN reader, which applies /100 at read via `_theta_batch_from_harmonic`. Do not apply /100 here.
4. **channel_slice**: if not None, apply `patches = patches[..., channel_slice]` **before** channel_scale. (For auto+cross it's None → all 10 channels.)
5. **channel_scale**: if not None, `patches = patches * scale` — **MULTIPLY**. ⚠️⚠️ The CNN reader DIVIDES (`maps_np / channel_scale`). The L1 reader MULTIPLIES (`patches * scale`), because the L1 `channel_scale = noise_sigma / channel_sigma` is constructed to amplify cross channels. Copying the CNN's divide would invert the normalization and destroy the result. Match `_load_harmonic_file` exactly (line ~390: `patches = patches * scale`), including the shape check `scale.shape == (patches.shape[-1],)`.
6. **split-slicing**: list shards with the SAME slicing as `.npz`. Reuse the already-fixed `_parse_harmonic_split_slice` + the `round(frac*n)` logic on the **sorted** shard list, so `train[70%:]` selects the same 1888 realizations as the `.npz` path (and as the CNN arms). The TFRecord shard stems are 1:1 with `.npz` stems (`{cosmo_id}_perm{perm}`), so sorted order is isomorphic.
7. **flip**: keep the L1's existing **numpy** flip (`_harmonic_random_flip`, per-patch LR/UD, train only, main thread). Do NOT adopt the CNN reader's *in-graph* flip — the L1 path is numpy/PyTorch, and numpy flip is correct here (the CNN moved flip in-graph only to unblock its JAX pipeline throughput). The bit-exact test (§5) runs with **flip OFF** to check the underlying data.
8. **channels constant**: 10 (`HARMONIC_CACHE_CHANNELS`). Patch shape (160,160,10), C-order.

**Parse reference (copy this logic, not the device-array part)** — CNN script `npe_cnn_nbody_tomo.py`:
- shard listing: `_list_harmonic_tfrecord_shards` (~line 1307)
- compression resolution: `_resolve_harmonic_tfrecord_compression` (~line 785) — reader compression MUST match the manifest (`NONE`); reuse this helper's approach.
- parse_fn: `build_harmonic_tfrecord_iterator` (~line 1345), specifically:
  - `patch = tf.reshape(tf.io.decode_raw(ex["patch"], tf.float32), (160,160,10))` (line ~1417)
  - `theta = tf.reshape(tf.io.decode_raw(ex["theta"], tf.float64), (6,))` (line ~1438)
- Example feature keys (from `build_harmonic_tfrecord.py`): `patch` (bytes), `theta` (bytes), `cosmo_id` (bytes), `perm` (int64), `patch_idx` (int64), `regime` (bytes), `split` (bytes).

**How to read one shard as a 48-patch block (the L1 contract):** read the shard with `tf.data.TFRecordDataset(shard_path, compression_type=...)`, `.map(parse_fn)`, `.batch(48)` (or collect all records), materialize to numpy → `maps (48,160,160,10)`, and take `theta` from the first record (all 48 share it; assert they're equal as a sanity check). Then apply slice → scale(multiply) → optional flip, and yield `(maps, theta, shard_path)`. Use `cosmo_id`/`perm` to derive a `path`-like identity string for logging and for the equivalence test's matching.

For throughput, you may parallelize shard reads (e.g. an interleave or a small thread/process pool of `tf.data` readers), but each **yield must remain a single realization's 48-patch block** in shard-sorted order (or any order — the L1 dataset computation doesn't require shard order, but the equivalence test matches by `(cosmo_id,perm)` identity so order-independence is fine).

---

## 4b. The DOMINANT speedup — compute the L1 datavector ONCE per arm and reuse it

The TFRecord format alone (§4) only ~3.3×'s the data-load. The bigger win is structural: **the L1 training+val datavector depends ONLY on the split (full vs 70/30) + channel config — NOT on the NDE seed or the observed perm.** Today the script recomputes the full ~50-min datavector on every run, so arm 1's 9 (seed×perm) runs redo identical work 9 times. Compute once, cache, reuse → 9× fewer datavector computations. Combined with TFRecord on that one computation: ~8 h/arm → ~1 h/arm.

Dependence (CORRECTED 2026-05-28):
- **perm** (`--harmonic-obs-perm`, ~line 2225) only selects the *observed* datapoint; the training set is unchanged. ✅ perm-independent.
- **seed** drives NDE init + jaxili's internal train/val split + the train flip RNG. The train datavector call passes `flip=True, rng=default_rng(seed+1001)` (~line 2499). ⚠️ **The L1 datavector is NOT flip-invariant** — measured: flip changes it ~10% (cross-channels + finite-patch boundary effects). So `flip=True` makes the datavector seed-dependent and blocks dedup. **Earlier "flip-invariant ⇒ no-op" reasoning was wrong.**

So dedup is **CONDITIONAL** on a one-time experiment (running in the TFRecord session as of 2026-05-28): does flip augmentation actually help the L1 **FoM3**? Inference is on the un-flipped obs in both L1 and CNN, so flip is purely train-time aug.
- **If flip=False ≈ flip=True in FoM3** → flip aug doesn't help L1 → set train `flip=False` (deterministic) and **enable dedup**.
- **If flip=True is meaningfully better** → **keep flip=True, forgo dedup** (don't trade accuracy for speed, and don't give L1 weaker aug than the CNN, which keeps flip=True). The TFRecord reader port alone still gives ~3.3×/run.

Implementation (in `npe_l1norm_cross_jaxili_nbody_tomo.py`) — only if the experiment greenlights flip=False:
1. Set the train datavector `flip=False`.
2. **Add a datavector disk cache** in `compute_l1_dataset_from_harmonic_cache`: key = hash of (regime, split, channel_slice, channel_scale, l1_nbins, n_scales, SNR ranges, subtract_coarse_mean, l1_implementation) — **NOT seed, NOT perm, NOT tfrecord-vs-npz** (data bit-identical either way). On hit, load `(theta, x)` `.npz` and skip; on miss, compute then save atomically. Suggested dir: `…/definitive_comparison/compressed/l1_<arm>/l1_datavector_<key>.npz`.
3. Then the 9 runs/arm self-deduplicate (first computes+caches, other 8 load → NDE + single-patch obs only).
4. Observed datavector always depends on perm — keep computing per run (one patch, fast).

If dedup is greenlit it's the dominant speedup (~8 h/arm → ~1 h/arm); if not, the format swap still helps per run. **The TFRecord reader port (§4) is unconditional and proceeds regardless of the experiment.**

---

## 5. Validation — the GATE (do before any relaunch)

Write `scripts/sbi/tests/test_l1_tfrecord_equivalence.py`. It must prove the **L1 datavector** is bit-identical between `.npz` and TFRecord paths (not just the patches — the whole point is the downstream L1 summary is unchanged).

Procedure:
1. Pick the first 5 sorted `nobnt/train` realizations.
2. Build the WLStatistics object exactly as the L1 script does (same n_scales, l1_nbins, SNR ranges, noise_sigma, channel_sigma/channel_scale). Use a fixed `channel_scale` (compute it once from the cache so both paths use the same array) and a fixed `channel_slice` (test both None and `slice(0,4)`).
3. **Path A (.npz)**: `iter_harmonic_examples(..., flip=False)` over those 5 realizations → maps → `compute_l1_batch(...)` → L1 datavectors `A (240, 2000)`; collect theta per realization. Tag by `(cosmo_id, perm)`.
4. **Path B (TFRecord)**: same with `tfrecord_dir=...`, `flip=False` → `B (240, 2000)`; same tags.
5. Match by tag; assert:
   - raw maps bit-identical (max abs diff **0.0**),
   - theta bit-identical (raw, H0≈68.5) (max abs diff 0.0),
   - **L1 datavectors equal**: target max abs diff 0.0. The wavelet transform runs on GPU (PyTorch); if there is tiny nondeterminism, allow `atol≤1e-5` and **document it explicitly with the measured value** — but first try to get exactly 0.0 (set torch deterministic flags; the inputs are identical so it should be).
6. Assert with `channel_slice=slice(0,4)` and with channel_scale applied — both bit-exact.
7. **Dedup viability = flip-FoM3 experiment (NOT a flip-invariance check)**: flip is known to change the L1 datavector ~10% (§4b), so do not assert datavector invariance. Instead, the gate for dedup is the FoM3 experiment (running in the TFRecord session): one flip=True vs one flip=False L1 auto+cross run, same seed, compare FoM3/σ. flip=False ⇒ dedup ON (train flip=False). flip=True better ⇒ dedup OFF (keep flip=True). Record the decision + numbers in the fiber.
8. **Cache round-trip** (only if dedup ON): compute → save → load the datavector `.npz`; assert loaded == computed bit-exact, and a second invocation hits the cache (no recompute) and yields the identical dataset.

Also re-run the existing CNN-side suite to confirm you didn't disturb shared code:
```
conda run -n jaxili python scripts/sbi/tests/test_tfrecord_equivalence.py   # must still PASS bit-exact
```
And a both-paths L1 smoke: one 50-epoch L1 auto+cross run on `.npz` vs one on TFRecord (same seed, same perm); best-val loss should be close (shuffle/flip RNG differ), and neither crashes.

**Gate**: do not relaunch production until the L1 datavector equivalence is 0.0 (or documented ≤1e-5) AND the CNN suite still passes.

---

## 6. Relaunch the L1 auto+cross arms (after the gate passes)

Two arms, 3 seeds × 3 perms each = 18 posteriors. Reference invocation (from the killed launch script's `run_l1_cross`, **adding** `--harmonic-tfrecord-dir`). Mirror the CNN flag pattern: **keep `--full-sphere-cross-cache <.npz>`** (still needed for the observed datapoint via `load_observed_from_harmonic_cache`, which stays on `.npz`) **AND add `--harmonic-tfrecord-dir <tfrecord>`** (redirects `iter_harmonic_examples` → train/val + calibration). Compression auto-read from the manifest.

Fixed pieces:
- `L1=scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py`
- `TFREC=/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid`
- `NPZ=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid` (still passed via `--full-sphere-cross-cache` for obs)
- `OUT=scripts/sbi/results/exploratory/definitive_comparison`
- Common L1 flags (verbatim from the prior runs): `--zero-mean-maps --map-kind nbody --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 --pca-components 0 --l1-min-snr -13 --l1-max-snr 13 --cross-snr-percentile 1.0 --batch-size 256 --learning-rate 0.0001 --npe-samples 100000 --no-wandb --cross-noise-model channel_empirical_global --epochs 50000`
- Arm 1 (full-train): `--nde-train-split train`, label `l1_autocross_fulltrain`.
- Arm 2 (70/30): `--nde-train-split train[70%:]`, label `l1_autocross_split70` (now works thanks to the split fix + your TFRecord port).
- Per run: `--harmonic-obs-perm <perm>`, `--seed <seed>`, `--cuda-visible-devices <gpu>`, `--save-dir $OUT/posteriors/<label>/train_<label>_s<seed>_p<perm>`, `--posterior-out $OUT/posteriors/<label>/<label>_s<seed>_p<perm>.npy`, `--figure-out .../<...>.pdf`.

⚠️ **Channel-sigma / SNR calibration with TFRecord**: confirm the L1 script still emits `cross_noise_model = channel_empirical_global` and a per-channel `channel_scale` table in stdout (NOT a warning/fallback) — see memory `feedback_l1_cross_must_use_harmonic_route.md`. If calibration reads TFRecord (because it goes through `iter_harmonic_examples`), the σ values must match the `.npz`-derived ones (they will, since data is bit-identical). Verify in the smoke log.

**Run plan (max out both GPUs):** arm 1 on GPU 0, arm 2 on GPU 1, in parallel; within each, loop seeds×perms. Smoke ONE (seed 41, perm 0) of each first; confirm posterior saved + `cross_noise_model=channel_empirical_global` in the log; only then launch the rest. Use `nohup`, log per run, write a status JSON.

**Expected anchors** (for sanity, from the audit; v2 channel-aware noise model):
- L1 auto+cross full-train pooled FoM3 historically ~38k at perm 0. Your 3-seed × 3-perm pooled may differ; **report what you measure**, don't force a number.
- L1 auto+cross 70/30 is **new** (never run) — expect ~24% below full-train if the split penalty matches auto-only, but this is the open question. Measure it.

---

## 7. After the L1 task: the still-blocked CNN work (separate, lower priority)

Only after the L1 auto+cross arms are done + validated + plots sent. These can also be handed to another fresh session.

- **⚠️ FIRST decide the compressor regime (global-shuffle caveat, §1a):** the two RealNVP compressors trained this session are on **`.npz`** (pool shuffle). The MAF compressors will be trained on **TFRecord** (global shuffle) for speed. Mixing regimes makes the RealNVP-vs-MAF companion comparison confounded. Andreas approved running everything under TFRecord, so the clean path is to **retrain the 2 RealNVP compressors on TFRecord** (now ~1.3 h each) so all 4 compressors share one regime, then run their NDE arms. Alternatively, accept the confound and document it. Surface this to Andreas before training the MAF compressors. (The `.npz`-trained RealNVP compressors + their diagnostics stay on disk as a reference either way.)
- **Phase 0a — `scripts/sbi/train_jaxili_from_compressed.py`** (NEW): standalone jaxili MAF NDE on a pre-compressed cache (`compressed/<arm>/cnn_{train,val,obs}.npz`). Trains, samples 100k, computes FoM3 + 2D FoM + marginal σ → `.npy`/`.meta.json`/`.fom.json`. Spec details in `~/.claude/plans/mighty-tumbling-sparrow.md` §0a. Unblocks CNN NDE arms (run on whichever compressor regime is chosen above).
- **Phase 0b — `--vmim-companion-backend {sbi_lens,maf}`** in `npe_cnn_nbody_tomo.py` (plan §0b): swap the VMIM companion flow to a MAF for the MAF-companion compressors (arms 7,8,10). Then train those 2 compressors (auto-only + auto+cross) — now FAST via `--harmonic-tfrecord-dir`. ⚠️ This DOES edit CNN compressor-training code (the shared `train_compressor_vmim` loop / companion construction), which the DLPack session also touched (now committed). Re-run the CNN-side tests (`test_tfrecord_*`) after, and a `.npz` compressor smoke, to confirm you didn't disturb the completed path.
- Then run all CNN NDE arms via 0a, and Phase C (final comparison table + corner overlays + SUMMARY.md).

---

## 8. Verification checklist (tick before declaring the L1 task done)

- [ ] Read `HARMONIC_TFRECORD_README.md`; `git status` clean; no edits to shared training-loop code (`_array_has_nan` already done by the other session).
- [ ] `iter_harmonic_examples` TFRecord branch added; channel_scale is **MULTIPLY**; theta is **raw** (H0 not /100); yields per-shard numpy 48-patch blocks **in sorted-stem order (no global shuffle)**; split-slicing reused; flip stays numpy.
- [ ] **Flip-FoM3 experiment resolved (§4b/§5.7)** — the dedup gate (NOT a flip-invariance check; flip changes the datavector ~10%). Decision + FoM3/σ recorded in the fiber.
- [ ] **Datavector reuse (§4b) — only if experiment greenlit flip=False**: train `flip=False`; disk cache keyed by (split, channel config, SNR…) NOT seed/perm/format; load-if-exists + round-trip bit-exact; 9 runs/arm self-deduplicate. (If flip=True kept: dedup skipped, documented; reader port still done.)
- [ ] GPU policy honored: **GPU 1 maxed, GPU 0 ≤45%**; GPUs 2/3 untouched.
- [ ] `test_l1_tfrecord_equivalence.py` written and PASSES: L1 datavector max abs diff 0.0 (or documented ≤1e-5), for slice=None and slice(0,4), with channel_scale.
- [ ] CNN suite (`test_tfrecord_equivalence.py`) still PASSES (you didn't disturb shared code).
- [ ] Both-paths L1 smoke (50 epochs) ran; `cross_noise_model=channel_empirical_global` confirmed in the TFRecord-path log.
- [ ] L1 auto+cross relaunched on TFRecord: 18 posteriors (2 arms × 3 seeds × 3 perms), 100k samples each, no NaN/Inf.
- [ ] FoM3 (per-seed, per-perm, pooled) + marginal σ computed for both arms; corner overlays made (full vs 70/30, and overlaid vs the L1 auto-only baselines to show the cross-channel gain).
- [ ] Plots sent to Andreas. Fiber Loop Status updated with the numbers.
- [ ] Speed sanity: L1 auto+cross on TFRecord runs materially faster than the killed `.npz` attempt (~55 min/inference). Report the it/s.

## 9. Key facts / gotchas recap

- **Read `scripts/sbi/HARMONIC_TFRECORD_README.md` first** — the completed CNN path is your parse reference.
- TFRecord path/compression: `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid`, compression **NONE** (auto-read from manifest; match it).
- **No global shuffle for L1** — yield shards in **sorted-stem order** (CNN's global shuffle is deliberate and CNN-only; copying it perturbs the L1 result, §1a).
- channel_scale: L1 **multiplies**, CNN **divides** — do not cross them.
- theta: L1 yields **raw** H0; CNN applies /100 at read. Don't apply /100 in the L1 reader.
- L1 maps → **numpy** (PyTorch), not JAX device arrays. Ignore the DLPack + in-graph-flip parts of the CNN reader.
- Keep `--full-sphere-cross-cache <.npz>` AND add `--harmonic-tfrecord-dir <tfrecord>` (obs stays on .npz).
- Don't edit shared training code (the `_array_has_nan` guard is already done by the other session).
- Don't rerun `launch_phase_a_overnight.sh` (broken arm 2 on .npz + redundant arm 3). Write fresh commands.
- Reuse `scripts/sbi/diagnose_compressor.py` for any compressor diagnostics.
- Env: `conda run -n jaxili python ...`. **GPU 1: max out. GPU 0: ≤45% only** (other tenant). GPUs 2,3 are others'. `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- CNN compressors trained this session are on `.npz` (pool shuffle) → not regime-matched to TFRecord; see §7 before the CNN arms.
- Don't claim success on "ran without error" — claim it on the bit-exact diff (0.0) and the measured FoM3/σ numbers.
