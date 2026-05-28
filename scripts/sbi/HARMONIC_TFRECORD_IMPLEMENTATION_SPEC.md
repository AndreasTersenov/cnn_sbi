# Implementation spec: TFRecord/tf.data path for the harmonic cross-map cache

**Audience**: a fresh Claude Code session with no prior context on this task.
**Goal**: eliminate the data-loading bottleneck in CNN compressor training on the
harmonic cross-map cache by replacing the GIL-bound `.npz` loader with a
TFRecord + `tf.data` pipeline — **without changing the numerical content of the
training data in any way**.

Read this entire document before writing any code. Do not skip the
"Non-negotiable invariants" or "Tests" sections. Do not take shortcuts. The
scientific validity of an ongoing experiment depends on the TFRecord path
producing **bit-identical** patches to the existing `.npz` path.

---

## 0. Why this exists (context)

The repo runs a controlled L1-vs-CNN weak-lensing inference comparison. CNN
compressors train on 10-channel harmonic cross-maps stored as per-realization
`.npz` files under:

```
/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid/
  manifest.json
  {nobnt,bnt}/{train,val,obs}/{cosmo_id}_perm{perm}.npz
```

Each `.npz` holds 48 patches. Training reads these through
`build_harmonic_batch_iterator` in `scripts/sbi/npe_cnn_nbody_tomo.py`, which
uses Python threads to load + zlib-decompress files. Measured behaviour:

- Auto+cross compressor training: **2.4 it/s**, GPU utilization **0–14%** (starving).
- Auto-only compressor training (TFDS / `tf.data`): **20.3 it/s**, GPU well-fed.
- Root cause: `numpy.load` holds the GIL during zlib decompression, so 4 loader
  threads deliver only ~1.6× throughput. The GPU waits ~0.4 s per batch.

The fix: store the same patches as **TFRecord shards on `/nas`** (49 TB free)
and read them with `tf.data` (C++ decompression, no GIL, AUTOTUNE prefetch),
matching the proven auto-only path. Expected: **~6–7×** (compute-bound ~15 it/s).

**`/nas` not `/mnt`**: `/mnt` (`/dev/md0`, native XFS) is 98% full (474 GB free).
`/nas` is a FUSE/mergerfs union, 49 TB free. TFRecord uses **sequential** reads
(FUSE-friendly) and the 764 GB RAM page-caches everything after epoch 1. Write
the TFRecord to `/nas/tersenov/harmonic_tfrecord/`.

---

## 1. Non-negotiable invariants (the data must not change)

The TFRecord path must reproduce, **exactly**, what the `.npz` path delivers to
the compressor. Any deviation introduces a confound into the experiment. The
following must be preserved bit-for-bit (float32) or set-for-set:

### 1.1 Patch payload
- Source array: `d["patches"]`, shape `(48, 160, 160, 10)`, dtype `float32`.
- Channel order is **fixed and must be preserved**: channels 0–3 are the 4 auto
  maps (tomographic bins 1,2,3,4 in order); channels 4–9 are the 6 cross maps
  in pair order `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)`. This ordering is set by
  `CROSS_PAIRS` in `build_full_sphere_cross_cache.py`. **Do not reorder.**
- Patches are already noised (shape noise baked in at cache-build time) and
  **demeaned per (patch, channel)**. The TFRecord stores them as-is.

### 1.2 theta  ⚠️ H₀/100 CONVERSION HAPPENS AT READ TIME
- Source: `d["theta"]`, shape `(6,)`, dtype `float64`, order
  `[Omega_m, sigma_8, w_0, H_0, n_s, Omega_b]`. The cache stores **raw H₀**
  (e.g. `68.5`), verified: a sample theta is `[0.4, 0.65, -1.58, 68.5, 1.02,
  0.0525]`.
- At read time, `_theta_batch_from_harmonic` (lines ~1035–1038) does TWO things:
  (1) broadcast to `(N_patches, 6)`, AND (2) **`theta_batch[:, 3] /= 100.0`** —
  converting H₀=68.5 → h₀=0.685. The consumer therefore sees h₀≈0.685, NOT 68.5.
- **The TFRecord reader MUST apply this same /100 on theta[3].** The cleanest,
  bug-proof way: store the **raw** theta (H₀=68.5) in the TFRecord, and at read
  time call the existing `_theta_batch_from_harmonic` (or replicate exactly:
  broadcast then divide index 3 by 100). Do not bake the /100 into the writer —
  keep the raw value on disk and convert at read, identical to the `.npz` path.
- Failing to divide H₀ by 100 trains the NDE on wrong cosmology and silently
  corrupts the experiment. The equivalence test (§4.1 step 7) compares the
  **post-conversion** theta (h₀≈0.685) on both paths.

### 1.3 channel_slice (auto-only-from-cache)
- When the CNN requests auto-only channels from the 10-channel cache, the `.npz`
  path applies `maps_np = maps_np[..., channel_slice]` where `channel_slice`
  is `slice(0, 4)` (the 4 auto channels). Applied **after** loading, **before**
  channel_scale.
- **Decision**: store all 10 channels in the TFRecord; apply the slice at read
  time, identical to the `.npz` path. Do NOT bake the slice into the TFRecord.

### 1.4 channel_scale (RMS normalization)
- When `--harmonic-normalize-input-channels` is set, the `.npz` path divides:
  `maps_np = maps_np / channel_scale`, where `channel_scale` is the per-channel
  RMS computed by `compute_harmonic_channel_rms` over the **compressor-train
  split** (line ~942). Applied **after** channel_slice.
- The TFRecord reader must apply the **same** `channel_scale` array, the same
  way (broadcast divide over the last axis), in the same order (slice → scale).
- **Critical ordering**: `.npz` path order is `load → slice → scale → (flip)`.
  The TFRecord path must be `parse → slice → scale → (flip)`. Slice before scale.

### 1.5 Split slicing (compressor/NDE disjointness)
- Splits may be `train`, `train[:70%]`, `train[70%:]`, `train[A%:B%]`, parsed by
  `_parse_harmonic_split_slice` (line ~767) and applied to the **sorted** file
  list by `_list_harmonic_cache_files` (line ~796): `files = sorted(...);
  files[lo:hi]` with `lo=round(low*n)`, `hi=round(high*n)`.
- The TFRecord path must produce the **same file subset** for a given split
  spec. Easiest correct approach: shard the TFRecord **one shard per source
  `.npz` file**, named so the sorted shard order matches the sorted `.npz`
  order, and apply the identical `round(frac*n)` slicing to the shard list.
  (See §3.2 for the sharding scheme that guarantees this.)

### 1.6 Zero-mean assertion
- The `.npz` path asserts `_assert_zero_mean_patches` (atol=1e-4, line ~881):
  `max over channels of |mean over (H,W) of patch|  ≤ 1e-4`. The TFRecord
  builder must assert this on every patch before writing, and the reader's
  smoke test must re-assert on a sample.

### 1.7 Random flip augmentation
- Train split only, the `.npz` path applies `_harmonic_random_flip` (line ~865):
  per-patch independent LR (`[:, :, ::-1, :]`) and UD (`[:, ::-1, :, :]`) flips,
  each with prob 0.5, drawn from the batch RNG.
- Flip is **stochastic augmentation**, so the TFRecord path need NOT reproduce
  the exact flip sequence. But it must implement the **same flip semantics**
  (independent LR/UD per patch, p=0.5, train only). **The equivalence test
  (§4.1) runs with flip DISABLED** so it checks the underlying data, not RNG.

### 1.8 Shuffling
- The `.npz` path shuffles via a pool ring buffer (cross-file mixing). The
  TFRecord path shuffles via `tf.data.Dataset.shuffle(buffer)`. Exact order
  need not match, but **the multiset of patches delivered per epoch must be
  identical** (every patch in the split appears, none duplicated within an
  epoch beyond what the pool path already does). The equivalence test checks
  the *set* of patches, not order.

### 1.9 Output format to the consumer
- `build_harmonic_batch_iterator` yields `dict` batches:
  `{"maps": (B, H, W, C) float32, "theta": (B, 6) float64}` where `C` is 10 or
  4 (post-slice). The TFRecord factory must yield the **same dict shape, keys,
  and dtypes**. `theta` must be float64 (cast after tf.data if needed).
- The iterator is **infinite** (the training loop calls `next()` per step). The
  TFRecord dataset must `.repeat()`.

---

## 2. Exact current code to study first

Open `scripts/sbi/npe_cnn_nbody_tomo.py` and read these, in order, before writing anything:

| Lines | Symbol | What to learn |
|-------|--------|---------------|
| ~881–891 | `_assert_zero_mean_patches` | the zero-mean check to replicate |
| ~865–878 | `_harmonic_random_flip` | flip semantics |
| ~767–793 | `_parse_harmonic_split_slice` | split spec grammar |
| ~796–830 | `_list_harmonic_cache_files` | sorted-then-sliced file selection |
| ~833–862 | `audit_harmonic_split_overlap` | disjointness audit (reuse for TFRecord) |
| ~894–922 | `iter_harmonic_examples` | the simple per-file iterator |
| ~942–1032 | `compute_harmonic_channel_rms` | how channel_scale is computed |
| ~1035–1040 | `_theta_batch_from_harmonic` | theta broadcast |
| ~1135–1252 | `build_harmonic_batch_iterator` | **the function being replaced** |
| ~3727–3750 | `_harmonic_dataset_iter_factory` | **the call site to branch** |

Also read `build_full_sphere_cross_cache.py` lines ~280–320 (the
`np.savez_compressed` call) to confirm the `.npz` key set and dtypes.

Constant: `HARMONIC_CACHE_CHANNELS = 10` (grep for it).

---

## 3. What to build

Three deliverables. Build and test them in the order given.

### 3.1 `scripts/sbi/build_harmonic_tfrecord.py` (NEW)

A one-time converter: reads the `.npz` cache, writes TFRecord shards to `/nas`.

**CLI:**
```
--cache-dir <path>        # the .npz cache root (full_sphere_cache_grid)
--out-dir <path>          # TFRecord root on /nas, e.g. /nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid
--regime {nobnt,bnt}      # which regime to convert (default: convert both)
--splits train,val,obs    # which splits (default all present)
--workers 24              # parallel conversion processes (see §3.1.1; ~24 is the
                          # I/O-bound sweet spot, 16 if the overnight job is live)
--compress NONE           # TFRecord compression: NONE (recommended) or GZIP —
                          # benchmark both per §3.1.1; reader must match this
--overwrite               # re-convert even if shard exists
```

**Sharding scheme (CRITICAL — preserves split slicing, §1.5):**
- **One TFRecord shard per source `.npz` file.** For source file
  `{cosmo_id}_perm{perm}.npz`, write shard `{cosmo_id}_perm{perm}.tfrecord`.
- This guarantees the sorted shard list is order-isomorphic to the sorted `.npz`
  list, so `_list_harmonic_cache_files`-style slicing (`sorted()[lo:hi]`)
  selects the *same* realizations.
- Each shard contains 48 `tf.train.Example` records, one per patch, **written
  in patch index order 0..47** (so patch identity is recoverable).

**Per-patch Example schema (use raw bytes for bit-exactness):**
```python
feature = {
    "patch":     bytes_feature(patch.astype(np.float32).tobytes()),  # (160,160,10) row-major
    "theta":     bytes_feature(theta.astype(np.float64).tobytes()),  # (6,)
    "cosmo_id":  bytes_feature(cosmo_id.encode("utf-8")),
    "perm":      int64_feature(int(perm)),
    "patch_idx": int64_feature(int(patch_idx)),       # 0..47
    "regime":    bytes_feature(regime.encode("utf-8")),
    "split":     bytes_feature(split.encode("utf-8")),
}
```
- **Do NOT use `float_list`** for the patch — it works but is 4× larger and
  slower. Raw `tobytes()` + `tf.io.decode_raw(..., tf.float32)` is bit-exact and
  compact.
- Store the **full 10 channels** (no slice, no scale applied at write time —
  those happen at read time per §1.3/§1.4).
- Before writing each patch, run `_assert_zero_mean_patches(patch[None], src)`
  (add a batch axis since the helper expects `(N,H,W,C)`). Abort the whole
  build on failure — a zero-mean violation means the source cache is corrupt.

**Provenance:** write `<out-dir>/<regime>/tfrecord_manifest.json` containing:
- source cache path + its `manifest.json` `args_sha256`
- per-split: list of shard filenames (sorted), shard count, total patch count
- channel count (10), patch shape (160,160,10), dtype float32
- compression used
- a content hash: for a deterministic sample (first 3 shards of train), the
  SHA256 of the concatenated raw patch bytes — used by the equivalence test.

**Parallelism:** use `multiprocessing.Pool(workers)`; each task converts one
`.npz` → one `.tfrecord`. Processes (not threads) so decompression parallelizes.

**Idempotency:** skip a shard if it exists and `--overwrite` not set; verify the
existing shard's record count is 48, else rewrite.

#### 3.1.1 Resource usage during conversion (read this — it governs `--workers` and `--compress`)

The conversion is **CPU + disk-I/O only. There is NO GPU work** — it's byte
reformatting (decompress `.npz` zlib → reserialize → write TFRecord). **Do not
attempt to use a GPU for the converter**; it would sit idle. The GPU payoff is
entirely on the training-read side (§3.2), which is the point of this whole task.

The converter is **disk-I/O-bound, not CPU-bound**:
- Reads ~304 GB total (≈9,200 `.npz` files: 6,293 train + 2,800 val + ~91 obs,
  ~33 MB each) from `/mnt` (`/dev/md0`, XFS RAID, **98% full**).
- Writes ~211 GB (GZIP) or ~304 GB (NONE) to `/nas` (FUSE/mergerfs, 49 TB free).
- zlib-decompress (read side) and optional GZIP-recompress (write side) are the
  CPU costs; both parallelize across processes (no GIL across processes).

Implications for the flags:
- **`--workers`**: 50 is the cap, but expect **diminishing returns past ~24**
  because the `/mnt` array's aggregate read throughput becomes the wall, not CPU.
  Start with `--workers 24`; only go to 50 if `iostat`/throughput shows headroom.
- **`--compress`**: benchmark **GZIP vs NONE on a small sample** (e.g. 50 files)
  and report build-time + on-disk size + a 200-batch read throughput for each.
  - **NONE** avoids the recompress CPU cost (faster build) AND avoids any
    decompress at train time (faster reads); costs ~304 GB on `/nas` (fine,
    49 TB free) and more `/nas` read bandwidth on the cold first epoch (then
    RAM-cached in the 764 GB page cache).
  - **GZIP** is ~211 GB and decompresses in C++ via tf.data (no GIL, so the
    training read is still fast), but doubles converter CPU and adds first-read
    decompress cost.
  - Default recommendation: **NONE**, given `/nas` space is abundant and RAM
    caches everything after epoch 1 — but let the benchmark decide. Record the
    chosen mode in the manifest; the reader's `compression_type` must match.

**Contention with the still-running overnight job (IMPORTANT):**
At the time of writing, an auto+cross compressor is still training and is
**reading the same `.npz` files** from `/mnt/.../full_sphere_cache_grid/`. A
50-process conversion reading those same files competes for the 98%-full array's
I/O and can slow both jobs. Before launching the full conversion:
1. Check whether that job is still alive (`nvidia-smi`, or look for
   `npe_cnn_nbody_tomo.py ... --compressor-steps 80000` in `ps`).
2. If it is: either **throttle to `--workers 16`** to leave I/O headroom, or
   wait for it to finish, or accept both run slower. The 764 GB page cache
   helps — once the `.npz` files are read once they're cached in RAM, so the
   second reader mostly hits cache. Do the small-sample benchmark first
   (low impact), then time the full run for a good moment.
3. The converter only **reads** `/mnt` and **writes** `/nas` — it never writes
   to `/mnt`, so it cannot corrupt or fill the nearly-full array.

Estimated wall time for the full `nobnt` conversion: ~15–25 min, dominated by
the `/mnt` cold read (~10–15 min) and `/nas` write, not CPU.

### 3.2 tf.data reader in `npe_cnn_nbody_tomo.py` (NEW function)

Add `build_harmonic_tfrecord_iterator(...)` mirroring the signature/return of
`build_harmonic_batch_iterator`:

```python
def build_harmonic_tfrecord_iterator(
    tfrecord_dir: Path,
    regime: str,
    split: str,            # may include [:70%] slicing
    batch_size: int,
    seed: int,
    flip: bool,
    max_realizations: int | None = None,
    channel_scale: np.ndarray | None = None,
    channel_slice: slice | None = None,
    shuffle_buffer: int = 4096,
    compression: str = "GZIP",
) -> Iterator[Dict[str, np.ndarray]]:
```

**Pipeline (order matters — must match §1.4):**
1. List shards: replicate `_list_harmonic_cache_files` selection but for
   `.tfrecord` files: `sorted(glob)`, apply the same `round(frac*n)` slice from
   `_parse_harmonic_split_slice`, then `[:max_realizations]`. **Reuse the
   existing parse helper** — import/refactor, don't re-derive the regex.
2. `tf.data.Dataset.from_tensor_slices(shard_paths)`; if train and shuffle,
   `.shuffle(len(shards), seed=seed)` the shard order each epoch.
3. `.interleave(lambda p: tf.data.TFRecordDataset(p, compression_type=compression),
   cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE,
   deterministic=not flip)` — parallel shard reads.
4. `.shuffle(shuffle_buffer, seed=seed)` for train (cross-shard patch mixing).
5. `.map(parse_fn, num_parallel_calls=AUTOTUNE)` where `parse_fn`:
   - `tf.io.parse_single_example` with the schema
   - `patch = tf.reshape(tf.io.decode_raw(feat["patch"], tf.float32), (160,160,10))`
   - `theta = tf.io.decode_raw(feat["theta"], tf.float64)` → shape (6,)
   - **slice**: if `channel_slice` is not None, `patch = patch[..., start:stop]`
     (translate the Python slice to static ints; step must be 1 — assert it).
   - **scale**: if `channel_scale` is not None, `patch = patch / tf.constant(
     channel_scale, tf.float32)` (post-slice channel_scale, so its length must
     equal the sliced channel count — assert).
6. `.batch(batch_size, drop_remainder=True)` (matches the `.npz` path which only
   yields full batches: `while cursor + batch_size <= n_pool`).
7. `.repeat()` (infinite).
8. `.prefetch(tf.data.AUTOTUNE)`.
9. Iterate: for each tf batch, convert to numpy, apply `_harmonic_random_flip`
   **in numpy** if `flip` (to use the identical flip implementation — do NOT
   reimplement flip in tf), then build theta via `_theta_batch_from_harmonic`
   semantics — i.e. the raw theta from the record (H₀=68.5) must have index 3
   divided by 100 (→ h₀=0.685) before yielding. Easiest: store raw theta in the
   record and call the existing helper, or apply `theta_np[:, 3] /= 100.0`
   explicitly. theta dtype float64. Yield `{"maps": maps_np, "theta": theta_np}`.
   **Do not forget the H₀/100 conversion (§1.2) — it is the most likely silent
   bug.**

**Why flip in numpy:** reuse `_harmonic_random_flip` verbatim so train-time
augmentation is identical in distribution to the `.npz` path. The flip RNG is
seeded from the same `seed` for reproducibility.

**channel_scale source:** unchanged — still computed by
`compute_harmonic_channel_rms` over the **compressor-train split of the `.npz`
cache** (not the TFRecord). The RMS is a property of the data, identical either
way; computing it from the `.npz` cache once (cached to disk) is fine and avoids
a second code path. Pass the resulting array into the TFRecord iterator.

### 3.3 Wire it into the call site

In `_harmonic_dataset_iter_factory` (line ~3727), branch on a new flag
`--harmonic-tfrecord-dir`:
- If set: call `build_harmonic_tfrecord_iterator(tfrecord_dir=..., ...)` with
  the **same** `split`, `seed`, `flip`, `max_realizations`, `channel_scale`,
  `channel_slice` arguments currently passed to `build_harmonic_batch_iterator`.
- If not set: unchanged (existing `.npz` path).

Add the arg in the parser near the other harmonic args (search
`--full-sphere-cross-cache`). Record it in the run meta.json (search where
`cnn_map_route` is written) as `harmonic_tfrecord_dir`.

Also: `audit_harmonic_split_overlap` must still run and still pass. Either point
it at the `.npz` cache (file names match shard stems, so overlap logic is
identical) or add a TFRecord-aware variant. Simplest: keep auditing the `.npz`
cache (the shard stems are 1:1 with `.npz` stems, so the audit result is valid).
**Document this choice in a code comment.**

---

## 4. Tests (all mandatory — do not skip)

Put test scripts in `scripts/sbi/tests/` (create if absent). Each must print a
clear PASS/FAIL and exit nonzero on failure.

### 4.1 Numerical equivalence test (THE critical one) — `test_tfrecord_equivalence.py`

This is the test that protects the experiment. It must prove the TFRecord path
delivers **bit-identical** patches to the `.npz` path.

**Procedure:**
1. Pick the first 5 sorted `.npz` files of `nobnt/train` (240 patches).
2. Build a tiny TFRecord from exactly those 5 files (call the builder on a temp
   out-dir, or build just those shards).
3. **Path A (.npz)**: load those 5 files directly with `numpy.load`, in sorted
   order, concatenate patches → `A_patches (240,160,160,10)`, and build
   `A_theta (240,6)` by calling `_theta_batch_from_harmonic(theta, 48)` per file
   (this broadcasts AND divides H₀ by 100, so `A_theta[:,3]≈0.685`). Tag each
   row with `(cosmo_id, perm, patch_idx)`.
4. **Path B (TFRecord)**: read the 5 shards with `tf.data.TFRecordDataset`
   (NO shuffle, deterministic, batch 48, flip=False), parse, and also tag each
   patch with `(cosmo_id, perm, patch_idx)` from the Example fields.
5. **Match by identity**: sort both by `(cosmo_id, perm, patch_idx)`. Assert the
   tag sequences are identical (same 240 patches present).
6. **Bit-exactness**: `np.array_equal(A_patches_sorted, B_patches_sorted)` must
   be **True** for the raw (unsliced, unscaled) patches. If not exactly equal,
   report `np.abs(A-B).max()` — it must be `0.0`. (Raw bytes round-trip is
   lossless; any nonzero diff is a bug in serialization/parsing/reshape.)
7. **theta**: `np.array_equal(A_theta_sorted, B_theta_sorted)` must be True.
8. **With channel_slice**: repeat the patch comparison applying
   `slice(0,4)` on both paths. Must be bit-identical.
9. **With channel_scale**: compute an arbitrary fixed `channel_scale` (e.g.
   `compute_harmonic_channel_rms` on those 5 files), apply `slice → divide` on
   both paths, assert `np.allclose(A, B, rtol=0, atol=0)` (must be exactly equal
   since it's the same float32 ops in the same order; if float32 op ordering
   differs, allow `atol=1e-6` and DOCUMENT why).
10. **Zero-mean**: assert `_assert_zero_mean_patches` passes on B's raw patches.

**Acceptance**: steps 6, 7, 8 bit-exact (max abs diff `0.0`); step 9 within the
documented tolerance. Any failure = STOP, do not proceed to production.

### 4.2 Split-slicing equivalence — `test_tfrecord_split.py`
- For `train`, `train[:70%]`, `train[70%:]`: assert the TFRecord shard selection
  (sorted `.tfrecord` stems, sliced) equals the `.npz` file selection (sorted
  `.npz` stems, sliced) — same stems, same count. Confirms §1.5.
- Run `audit_harmonic_split_overlap` for `train[:70%]` vs `train[70%:]` and
  assert `overlap_count == 0`.

### 4.3 Epoch-completeness (set equality) — `test_tfrecord_epoch.py`
- For a small split (e.g. `val` limited to 10 shards = 480 patches), iterate the
  TFRecord dataset for exactly one epoch (no repeat, no shuffle) and collect the
  `(cosmo_id, perm, patch_idx)` tags. Assert the set equals the `.npz` set:
  every patch present exactly once. Confirms §1.8.

### 4.4 Output-contract test — `test_tfrecord_contract.py`
- One batch from `build_harmonic_tfrecord_iterator` (batch_size=128, train,
  flip=True): assert `batch["maps"].shape == (128,160,160,C)`,
  `batch["maps"].dtype == float32`, `batch["theta"].shape == (128,6)`,
  `batch["theta"].dtype == float64`. With `channel_slice=slice(0,4)`: `C==4`.
  Without: `C==10`.

### 4.5 Throughput sanity (not correctness, but the whole point) — `test_tfrecord_throughput.py`
- Time 200 batches (batch 128) from the TFRecord iterator vs the `.npz`
  iterator on `nobnt/train`. Print patches/s for both. Expect TFRecord ≥ 3×
  the `.npz` rate. (Informational; a <3× result means the pipeline isn't
  configured right — likely missing `interleave`/`prefetch`/AUTOTUNE.)

### 4.6 End-to-end smoke — `test_tfrecord_smoke_train.py` (or reuse the launch smoke)
- Run a 500-step compressor training with `--harmonic-tfrecord-dir` set, on
  GPU 1 (or whichever is free), and assert: no crash, loss decreases, best-val
  checkpoint saved, compressed `cnn_train.npz`/`cnn_obs.npz` produced. Compare
  the final `best_val_loss` to a 500-step run on the `.npz` path with the
  **same seed** — they should be **close** (not identical, because shuffle order
  and flip RNG differ, but within a few % — a large divergence signals a data
  bug the equivalence test missed).

---

## 5. Build & test order (do not reorder)

1. Read all code in §2. Confirm the `.npz` schema and the channel order.
2. Write `build_harmonic_tfrecord.py`. Convert ONLY the first 5 `nobnt/train`
   files to a temp dir for now (a `--limit-files 5` dev flag helps).
3. Write `test_tfrecord_equivalence.py` (§4.1). **It must pass bit-exact before
   anything else.** Iterate on the builder/parser until step 6 max-abs-diff is
   exactly 0.0. This is the gate.
4. Write the reader `build_harmonic_tfrecord_iterator` + wire the flag (§3.2,
   §3.3).
5. Run §4.2, §4.3, §4.4. All must pass.
6. Run §4.5 throughput. Confirm ≥3×.
6b. **Compression benchmark (§3.1.1):** convert ~50 files with `--compress NONE`
   and again with `--compress GZIP`; record build-time, on-disk size, and
   200-batch read throughput for each. Pick the winner (expected: NONE). Use it
   for the full conversion and set the reader's `compression_type` to match.
7. Full conversion: first check the overnight auto+cross job isn't still reading
   `/mnt` (`ps | grep 'compressor-steps 80000'`); pick `--workers` per §3.1.1
   (24 if clear, 16 if the job is live). Run the builder on the **entire**
   `nobnt` regime (train+val+obs), writing to
   `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid/`. Verify the
   manifest and shard counts (train should be 6293 shards).
8. Run §4.6 end-to-end smoke (500 steps) on the full TFRecord. Compare best_val
   to the `.npz` path.
9. Only then: hand back for production use (the campaign launcher will add
   `--harmonic-tfrecord-dir`).

---

## 6. Pitfalls & gotchas (read before coding)

- **Endianness / dtype in `decode_raw`**: `np.float32.tobytes()` is little-endian
  on this x86 host; `tf.io.decode_raw(bytes, tf.float32)` reads little-endian by
  default. theta is float64 — use `tf.float64` in its `decode_raw`. Mixing these
  up gives garbage, not an error — the equivalence test catches it.
- **Reshape order**: `tobytes()` is C-order (row-major). `tf.reshape` is also
  C-order. Reshape to `(160,160,10)`. Do not transpose.
- **channel_slice with step**: the `.npz` path supports arbitrary slices but in
  practice it's `slice(0,4)` or `None`. Assert `channel_slice.step in (None,1)`
  and translate to `patch[..., start:stop]`. Fail loudly on step≠1.
- **channel_scale length after slice**: if slice is `[:4]`, `channel_scale` must
  have length 4 (it's computed with the same `channel_slice`, see §3.2). Assert
  `len(channel_scale) == sliced_channel_count` in the reader.
- **drop_remainder=True**: required to match the `.npz` path's full-batch-only
  behaviour and to give the compressor a static batch shape.
- **`.repeat()` placement**: after `.batch`, before `.prefetch`. The training
  loop expects an infinite iterator.
- **float32 op ordering in scale**: `patch / channel_scale` in numpy vs tf should
  be identical for the same float32 inputs (IEEE 754 division is deterministic).
  If §4.1 step 9 shows a tiny nonzero diff, it's because the unscaled patches
  already differed (a real bug) — chase that, don't paper over it with tolerance.
- **GZIP vs NONE compression**: GZIP shards are ~the same size as `.npz` and
  decompress in C++ (no GIL) — use GZIP. NONE is bigger but marginally faster to
  read; not worth the extra `/nas` space. The reader's `compression_type` must
  match the builder's `--compress`.
- **Do not apply flip in the tf graph**: reuse `_harmonic_random_flip` in numpy
  on the batch after materializing it, so augmentation matches the `.npz` path.
- **`/nas` is FUSE**: writing 6293 shards in parallel is fine, but list
  operations can be slow — cache the sorted shard list, don't re-glob per batch.
- **Do not modify** `build_harmonic_batch_iterator` or any `.npz`-path code. The
  TFRecord path is purely additive (a new branch behind a flag), so the existing
  comparison stays reproducible and we can A/B the two paths.

---

## 7. Acceptance criteria (definition of done)

- [ ] §4.1 equivalence: raw patches & theta **bit-identical** (max abs diff 0.0);
      sliced & scaled within documented tolerance (target 0.0).
- [ ] §4.2 split slicing: TFRecord shard selection == `.npz` selection; 70/30
      overlap == 0.
- [ ] §4.3 epoch completeness: patch set identical.
- [ ] §4.4 output contract: shapes/dtypes/keys correct for sliced and unsliced.
- [ ] §4.5 throughput: TFRecord ≥ 3× `.npz` patches/s.
- [ ] §4.6 end-to-end: 500-step training runs, best_val close to `.npz` path.
- [ ] Full `nobnt` regime converted to `/nas`, manifest verified (6293 train
      shards), zero-mean assertion passed on every patch during build.
- [ ] `--harmonic-tfrecord-dir` flag wired, recorded in run meta.json, `.npz`
      path untouched and still default.
- [ ] No edits to existing `.npz`-path functions; TFRecord path is additive.

When all boxes are checked, report back with: the equivalence-test max-abs-diff
(should be 0.0), the throughput numbers, and the smoke best_val comparison.
Do not declare success on "ran without error" — declare it on the numbers.
```
