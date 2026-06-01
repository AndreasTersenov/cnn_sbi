# Plan — proper fast CNN data path for auto+cross (TFDS / Grain)

**Date:** 2026-05-29 · **Branch:** `autoresearch/cnn-auto-push-18-20-2026` · **Status:** plan for sign-off, no code yet.

> Supersedes the earlier "thin raw-bytes numpy loader" idea in this file. That was
> still a hack bolted onto the wrong foundation (the per-realization `.npz` cache /
> hand-rolled TFRecord). The deeper finding below points at the actual fix: make the
> 10-channel data a *proper TFDS dataset* like auto-only, and load it with a standard
> loader. The benchmark harness (`scripts/sbi/tests/bench_cnn_loader.py`) is kept as a
> measurement tool for Phases 0/2.

---

## 1. Honest diagnosis (verified by reading the code + the run, not assumed)

**Auto-only is fast because it uses the standard path.** `npe_cnn_nbody_tomo.py:2528`:
```python
ds = tfds.load(tfds_name, split=split)
ds = ds.repeat().shuffle(buf).map(aug, num_parallel_calls=AUTOTUNE).batch(bs).prefetch(AUTOTUNE)
return iter(tfds.as_numpy(ds))     # plain numpy handoff — no DLPack, no interleave, no thread budget
```
TFDS serialized the 4-channel maps once into its standard sharded format; `tfds.load`
gives an optimized pipeline; `tfds.as_numpy` yields numpy that JAX moves to GPU. Clean.

**Auto+cross is slow/fragile because it bypasses TFDS entirely.** It uses a custom
per-realization `.npz` cache (zlib → GIL-bound → ~2.4 it/s), and then a hand-rolled
`build_harmonic_tfrecord_iterator` (`:1389`) doing `interleave → shuffle → parse →
in-graph flip → DLPack→GPU → thread budget`. Measured 2026-05-29 (bench harness, GPU 0,
node load ~7 on 128 cores, no cgroup cap): **~1 it/s with 549 threads pinned at one core
while ~120 cores sat idle** — i.e. lock/thread *thrash*, not CPU starvation. The DLPack
handoff, the in-graph flip, and the thread-budget knob **all exist only to prop up this
hand-rolled choice.**

**Root cause:** we never promoted the 10-channel data to a TFDS dataset; we dumped it to
`.npz` and bolted a custom tf.data pipeline on top. Web research confirms the tf.data
AUTOTUNE thread behavior is a *documented, version-flaky pathology* (`cpu_budget` /
threading options unreliable across TF versions) — which is exactly why every
thread-budget patch we tried backfired. The standard `tfds.load` path (auto-only) does
*not* hit this; our custom `interleave`+DLPack pipeline did.

**Why the 10-channel data CAN be a TFDS dataset (the key unlock).** It is structurally
identical to auto-only — 48 patches/realization of `[H,W,C]` float32 + `theta[6]`, just
`C=10` (4 auto + 6 cross) instead of 4. The only extra work is computing the 6 cross
maps, which `build_full_sphere_cross_cache.py` already does: load 4 auto maps → **add
shape noise on the sphere** → SHT → 6 element-wise `a_lm` cross-products (Zürcher 2022) →
iSHT all 10 → GnomonicProj into 48 patches. Noise is *baked in before the cross-product*
because the cross channels are nonlinear in the noisy `a_lm` (you cannot add cross-channel
noise per-batch the way auto-only does). Cost ~30–50 s/job; `Pool(50)` ⇒ a one-time
precompute (hours), exactly the kind of upfront investment that building the auto-only
TFDS already represented.

## 2. The proper approach (standard, non-hacky)

**Build the auto+cross data ONCE as a proper TFDS dataset, then train via a standard
loader** — dropping the `.npz` cache, the custom TFRecord reader, DLPack, and the thread
budget entirely. Build in **ArrayRecord** format so either loader below can read it.

Two standard loaders, decided by benchmark (not vibes):

- **(A) TFDS + tf.data** (`tfds.load` / `tfds.data_source`), identical class to auto-only.
  Lowest risk, proven in-repo, reuses the existing `_dataset_iter`.
- **(B) TFDS(ArrayRecord) + Grain** — Google's JAX-native loader (production-stable
  v0.2.16, Feb 2026; used by MaxText/Gemma). Uses worker **processes**
  (`mp_prefetch(MultiprocessingOptions(num_workers=…))`), *not threads* → structurally
  immune to the 549-thread thrash; **deterministic true global shuffle** via ArrayRecord
  random access; no full-TensorFlow dependency at train time; scales to our 50 CPUs as
  `num_workers`. The modern standard the community has moved to. Typical shape:
  ```python
  src = grain.sources.ArrayRecordDataSource(paths)
  ds = (grain.MapDataset.source(src).shuffle(seed).map(parse_normalize_flip)
        .to_iter_dataset().batch(bs, drop_remainder=True)
        .mp_prefetch(grain.multiprocessing.MultiprocessingOptions(num_workers=16)))
  it = iter(ds)
  ```

Both share the **same data build**; the loader is a swap.

## 3. Science preservation (non-negotiable gates)

1. **Noise baked in on the sphere** before the harmonic cross-product (unavoidable; matches
   current cache). Same seed scheme (`noise_seed_base + 100*cosmo_idx + perm`) ⇒ maps
   reproducible/identical to the current `.npz`.
2. **Per-channel normalization** (cross channels ~10⁴× smaller): a fixed per-channel scale
   computed once from train, applied in the pipeline `map` (matches
   `--harmonic-normalize-input-channels`).
3. **Channel order `[4 auto, 6 cross]`** preserved exactly.
4. **Flips-only augmentation** in-pipeline (no per-batch noise for cross — matches current).
5. **Global shuffle** (matches auto-only). Re-run BOTH arms being compared under the new
   path; never compare a new-path CNN to an old-path one (existing caveat).
6. **HARD GATE — bit-exact** map data + `theta` (post `H0/100`) vs the current `.npz`
   cache (max abs diff 0.0), reusing the equivalence-test discipline already in
   `tests/test_tfrecord_equivalence.py`.
7. **END-TO-END GATE** — a cross arm's FoM3 / marginal σ under the new path match the
   current pipeline within seed noise (no science regression). FoM3 is fragile, so check
   2D areas + marginals too ([[feedback-fom3-fragile-use-2d-areas]]).

## 4. Phases

- **Phase 0 — Ground truth (harness, short).** With `bench_cnn_loader.py`, *measure* (not
  assume): auto-only TFDS it/s + thread count (confirm the standard path is fast AND
  low-thread), and re-measure the current custom auto+cross path (confirm the 549-thread
  thrash, ≥3 runs, load/threads stamped). Add a `tfds_auto` candidate. This is the
  measured baseline the rebuild must beat. *(Needs a free-ish GPU; coordinate vs the L1
  campaign on GPU 1.)*
- **Phase 1 — Build the proper dataset (the upfront work).** Extend
  `tf_dataset_nbody_tomo.py` with a 10-channel builder (config or new
  `NbodyCosmogridDatasetTomoCross`) whose `_generate_examples` reuses the cross-map
  computation from `build_full_sphere_cross_cache.py`. Build with
  `tfds build … --file_format=array_record`. **Bit-exact-validate** vs the `.npz` cache on
  a subset. Confirm `/nas` free space (~300 GB nobnt). Decide BNT scope.
- **Phase 2 — Clean loader + benchmark A vs B.** Implement the standard tf.data loader
  (reuse `_dataset_iter`) and a Grain loader; benchmark both vs the old path with the
  harness (≥3 runs, stamped). Pick by numbers.
- **Phase 3 — Integrate + science validation + delete the cruft.** Wire the chosen loader
  as the auto+cross route in `npe_cnn`, apply normalization + flips in-pipeline, re-run a
  cross arm (single-GPU), confirm FoM3/marginals match current within noise. **Only then**
  delete the dead machinery (`build_harmonic_tfrecord_iterator`, DLPack handoff, thread
  budget, `build_harmonic_tfrecord.py`, the `.npz` loader if fully replaced). Throughout
  Phases 0–3, use multi-GPU meaning **(i)** (independent jobs per card) for campaign
  throughput — it's free.
- **Phase 4 — Data-parallel single run (multi-GPU meaning (ii)), if chosen.** Grain shards
  across cuda 0,1; JAX data-parallel for the compressor with an **all-gather** of
  summaries+theta so the VMIM MI estimate stays full-batch. Science gate: FoM3/marginals
  match the single-GPU VMIM result. Kept separate from the data-path rebuild so a
  distributed-loss change never confounds the loader change.

## 5. Decisions (locked 2026-05-29)

1. **Loader:** build in **ArrayRecord** (serves both), **benchmark tf.data vs Grain** in
   Phase 2, **lean Grain** unless tf.data is clearly faster.
2. **Scope:** **nobnt only** (~300 GB).
3. **Isolate:** fix **auto+cross only** for now; leave the working auto-only TFDS path
   untouched.
4. **Multi-GPU: single-GPU for now** (decided 2026-05-29). Each training run uses one
   card. We may still use both A100s for *independent* concurrent jobs (meaning (i) in
   §5a) for campaign throughput — that's free. The data-parallel single-run case
   (meaning (ii), with VMIM all-gather) is **deferred to a later Phase 4**, not now.
5. **Noise diversity:** default to one baked realization per (cosmo,perm) for parity with
   the current science (revisit only if augmentation proves limiting).

### 5a. Multi-GPU — one honest distinction to confirm

"Use both A100s" has two very different meanings, with very different cost/risk:

- **(i) Campaign-level parallelism** — run *independent* jobs concurrently, one per A100
  (e.g. two seeds, or L1 on one card + CNN on the other). **Trivial and zero science
  risk**: the orchestrators already take `--gpus 0,1`; each job is a normal single-GPU
  run. Gives ~2× *campaign* throughput.
- **(ii) Data-parallel a single training run** across both A100s. Grain shards the data
  cleanly and JAX data-parallel is standard — **but our compressor uses VMIM, whose loss
  couples samples *within a batch*** (the in-batch samples act as the contrastive/negative
  set for the mutual-information estimate). Splitting a 128-batch into 64+64 across two
  devices changes the MI estimator **unless we all-gather** the summaries+theta so each
  device computes the full-batch MI. So (ii) needs an explicit all-gather and a science
  gate (FoM3/marginals must match the single-GPU VMIM), on top of the data-path work.

**My recommendation:** get the clean data path + single-GPU science validation done first
(Phases 0–3), using **(i)** for throughput (it's free and already supported). Then add
**(ii)** as a Phase 4 with the VMIM all-gather + its own validation — rather than
entangling a science-sensitive distributed-loss change with the data-path rebuild. If you
specifically want one *run* to use both cards, we do (ii), just with eyes open on the
all-gather + extra validation. **Confirm which meaning you want.**

## 6. Not yet verified (no guessing — Phase-0/1 checks)

- Auto-only's actual it/s + thread count on our hardware (documented ~20; will measure).
- Exact bit-exact reproducibility of the cross-map compute through a TFDS generator
  (will test vs `.npz`).
- `/nas` free space for the ArrayRecord build.
- Grain ↔ our jax/jax-cosmo/sbi_lens jitted `update` + jax version (should be fine — plain
  numpy batches — but will smoke-test).

## 7a. Verified groundwork (2026-05-29, in the `jaxili` env)

- **Env supports the build:** `tensorflow 2.18.0`, `tensorflow_datasets 4.9.9`,
  `array_record 0.8.3` all present; tfds file formats include `array_record`. So
  `tfds build … --file_format=array_record` works today.
- **`grain` is NOT installed** → `pip install grain` needed, but only for Phase 2 (the
  loader benchmark). Phase 1 (the build) needs only TFDS+array_record, which are present.
- **Disk:** `/nas` has 48 T free; the current `.npz` nobnt cache is 305 G — fits easily.
- **Cross-map compute (one source of truth) is fully mapped** in
  `build_full_sphere_cross_cache.py`: `CROSS_PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))`,
  `N_AUTO=4`, noise `seed = noise_seed_base + 100*cosmo_idx + perm` →
  `np.random.default_rng(seed).normal(0, per_pixel_std)`, `map2alm(iter=0)` →
  `alm[i]*alm[j]` → `alm2map` → `_patch_one_realization` (GnomonicProj + per-patch
  demean). theta is float64 `[Om,s8,w0,H0,ns,Ob]` with **H0 left un-scaled** (the `/100`
  is applied later in training preprocessing — same as auto-only). The 48 centers come
  from `_build_non_overlapping_centers` **imported from the auto-only builder**, so the
  patch geometry is already shared with the auto-only TFDS.

## 7b. Concrete Phase-1 design (for sign-off)

- **Refactor for one source of truth:** extract the pure compute (noisy maps → 10-ch
  patches, steps 2–7) out of `_worker` in `build_full_sphere_cross_cache.py` into a
  function `compute_cross_patches(noiseless, cosmo_idx, perm, regime, cfg, centers) ->
  patches[48,H,W,10]`. Call it from **both** the existing `.npz` builder (output must stay
  byte-identical — validated) **and** the new TFDS builder. No duplicated math.
- **New builder** `tf_dataset_nbody_tomo_cross.py`: a `GeneratorBasedBuilder` mirroring
  the auto-only one, FeaturesDict `{"map": Tensor[H,W,10] f32, "theta": Tensor[6] f32}`,
  config `grid_20deg_160px_nonoverlap48` (nside 512, lmax, 48 non-overlap centers), same
  train/test split convention. `_generate_examples` loops cosmo→perm, loads the 4
  noiseless maps, calls `compute_cross_patches`, and **yields one example per patch**
  (`f"{cosmo}-{perm}-{k}"`) — matching auto-only's per-patch granularity and global
  shuffle.
- **Build** nobnt only, `--file_format=array_record`, to `/nas/tersenov/...`.
- **HARD bit-exact gate:** for a handful of (cosmo,perm), assert TFDS patches == the
  existing `.npz` patches (max abs diff 0.0) and theta matches. Reuse the
  `tests/test_tfrecord_equivalence.py` discipline. This is the gate that proves the
  builder preserves the science before we spend hours on the full build.

## 7c. Progress (2026-05-29)

**Phase 1 implemented + validated (bit-exact).**
- Refactored `build_full_sphere_cross_cache.py`: cross-map compute extracted into shared
  `compute_noisy_alms` / `cross_patches_from_alms` / `compute_cross_patches` (one source of
  truth). Behavior-preserving — `tests/validate_cross_compute_refactor.py` recomputes a
  realization → **max abs diff 0.0** vs the `.npz` cache.
- `tf_dataset_nbody_tomo_cross.py`: TFDS `GeneratorBasedBuilder` (10-ch `map_nbody` + theta
  + provenance). It **reserializes the validated `.npz` cache** (one example/patch) rather
  than recompute. `tests/test_tfds_cross_equivalence.py` builds a subset in ArrayRecord →
  **map diff 0.0, theta diff 0.0** across train/val/obs.
- `build_cross_tfds_dataset.py`: programmatic build wrapper (`--file-format`, `--cosmo-limit`).

**Findings that corrected the plan:**
- **ArrayRecord ⊥ `as_dataset()`** — must read via `as_data_source()` (Grain's random-access
  API). The vanilla auto-only `tfds.load().as_dataset()` tf.data path needs **TFRecord**.
- **Serial TFDS build ≈ 17–20 examples/s** (bottlenecked decompressing the zlib `.npz` on one
  core) → full grid (~440k examples) ≈ **6–7 h/format serially**, NOT the ~1 h I guessed
  earlier (an unmeasured estimate — corrected by the gate's measured rate). `apache_beam`
  (TFDS's parallel build) is not installed; a custom `mp.Pool(50)` converter could
  parallelize the zlib decode if we want the full build faster.
- `grain 0.2.16` installed (dry-run confirmed it adds only the grain wheel — no dep
  changes, safe alongside the running L1 campaign).

**In flight:** benchmark subset build (20 cosmologies/split, ArrayRecord) →
`/nas/tersenov/tfds_cross_arrayrecord_subset20` (~18 min). Next: Phase 2 — write the Grain
loader, benchmark Grain-on-ArrayRecord vs the old custom path on the subset (confirm fast +
low-thread) BEFORE committing to the full ~7 h build.

## 7d. Phase 2 results — the decisive benchmark (2026-05-29)

The decisive 3-way comparison (back-to-back, GPU 0):

| candidate | median it/s | p10–p90 | GPU util | load (pre→post) |
|---|---|---|---|---|
| **auto_tfds** (4-ch, std `tfds.load` tf.data) | **30.7** | 29.3–31.6 | 37% | 8.9→20.8 |
| **tfdata_cross** (10-ch, std `tfds.load` tf.data on TFRecord) | **16.8** | 14.4–17.7 | 17% | 20.8→**19.9** (stable) |
| grain_w32 (10-ch, Grain on ArrayRecord) | 6.7 | 5.3–8.2 (variable) | **0%** | 19.9→**45.2** |

**The decision: adopt `tfdata_cross`, drop Grain.** Standard `tfds.load` + tf.data on a
TFRecord cross dataset (mirroring auto-only's mechanism) is **2.5× faster than Grain
under the same contention AND keeps node load stable**, while Grain's 32 worker
processes drove load 20→45 and starved the GPU. **For this workload, Grain's
mp_prefetch overhead + inter-process shared-memory handoff is heavier than tf.data's
in-process AUTOTUNE.** "The clean answer is also the simplest one" — replicate exactly
what auto-only does, just on the 10-channel data.

`tfdata_cross` ≈ 17 it/s is at/near the **intrinsic 10-channel ceiling** (naive
2.5×-channel scaling predicts ~12; we beat that, likely because per-step host syncs
cap things before pure transfer does). "As fast as auto-only" was a wrong implicit
target — 10 channels can't match 4. "At the cross ceiling with low overhead" is the
right one, and `tfdata_cross` hits it.

### Two findings that corrected the path

1. **The mp.Pool build speedup estimate was wrong.** I guessed "~7 h → ~20 min" with
   `mp.Pool(50)`. Measurement showed the bottleneck is the TFDS *serializer* (~20
   examples/s), not the zlib decode. Parallel decode → workers wait idle for the
   writer. Net speedup ~15% (with `DISABLE_SHUFFLING=True` saving a second pass).
   Full serial build is still ~5–7 h.
2. **`apache_beam` (TFDS's standard parallel-build mechanism) is too risky to install
   here.** Dry-run shows it would upgrade **protobuf 5.29.3 → 6.33.6** plus pull
   dozens of heavy deps (grpcio, cryptography, betterproto, envoy_data_plane…). A
   protobuf major-version bump would likely break TensorFlow 2.18 in the shared env,
   which would silently break the **running L1 campaign**. Accepting ~6 h serial
   build instead.

### Phase 3 status (in flight)

- **Adopted:** `tfds_cross_tfdata_loader.build_tfds_tfdata_iterator` (TFRecord cross,
  standard tf.data). Wired into `npe_cnn` behind `--cross-tfdata-dir`.
- **Full TFRecord build LAUNCHED 2026-05-29 22:24 UTC** to
  `/nas/tersenov/tfds_cross_tfrecord_full` (PID 3227124, expected wall ~5–7 h, serial
  writer). `DISABLE_SHUFFLING=True` + `CROSS_TFDS_BUILD_WORKERS=50` (the parallel
  decode saves ~15%, mostly idle).
- **Confirmation (3× `tfdata_cross` runs)** in a low-load window: in flight.
- **Pending Phase 3 work:**
  - **Science validation**: run a real cross arm end-to-end with `tfdata_cross`,
    confirm FoM3 + marginals match the legacy `.npz` pipeline within seed noise.
  - **`compress_dataset` refactor**: the NDE-prep currently reads `.npz` serially
    (~1 h after training). Re-point it at the TFRecord dataset so the *full pipeline*
    is fast, not just the training loop.
  - **Deletion** (only after validation): `grain_loader.py`, the `grain` pip
    dependency, `build_harmonic_tfrecord_iterator` + the DLPack handoff + the
    thread-budget machinery (`_resolve_cnn_cpu_threads`, the BLAS-env block, the TF
    intra/inter setters), the `--grain-tfds-dir` / `--harmonic-tfrecord-dir` flags
    and their factory branches. Likely keep `.npz` loader for read-time
    compatibility (other paths use it) but remove from the CNN's hot path.

## 7. Files in play

- `scripts/sbi/tf_dataset_nbody_tomo.py` — the auto-only TFDS builder (the template to extend).
- `scripts/sbi/build_full_sphere_cross_cache.py` — the cross-map computation to reuse in the generator.
- `scripts/sbi/npe_cnn_nbody_tomo.py` — `_dataset_iter` (`:2528`, standard path), the harmonic route (`:2542`, `:4057`, `:4069`) to be replaced.
- `scripts/sbi/tests/bench_cnn_loader.py` — measurement harness (Phases 0/2).
- `scripts/sbi/HARMONIC_TFRECORD_README.md` — documents the to-be-retired hand-rolled path.
