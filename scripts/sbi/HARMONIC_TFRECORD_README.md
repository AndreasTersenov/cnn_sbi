# Harmonic cross-map TFRecord training path

A faster data path for **CNN compressor training** on the 10-channel harmonic
cross-map cache. It replaces the GIL-bound per-realization `.npz` loader with a
TFRecord + `tf.data` pipeline, **~7.4× faster** (2.4 → ~17 it/s on the auto+cross
`plain` compressor) while delivering **numerically identical** training data.

Implements `scripts/sbi/HARMONIC_TFRECORD_IMPLEMENTATION_SPEC.md` (read that for
the full invariants). This file is the short runbook + design note.

## How to use

**1. Build the shards (one-time, CPU/IO only — touches no GPU):**

```bash
conda run -n jaxili python scripts/sbi/build_harmonic_tfrecord.py \
  --cache-dir scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid \
  --out-dir   /nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid \
  --regime nobnt --splits train,val,obs --workers 16 --compress NONE
```

The full `nobnt` regime is **already built** at the `--out-dir` above:
9184 shards (train 6293 / val 2800 / obs 91), 421 GB, `NONE` compression.
One shard per source `.npz`, identical stem, 48 patches each.

**2. Train against it:** add one flag to the usual CNN command —

```bash
... --full-sphere-cross-cache <.npz cache> \
    --harmonic-tfrecord-dir /nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid
```

`--full-sphere-cross-cache` is **still required** — the `.npz` cache is used for
channel-RMS normalization, the observed datapoint, and the split-overlap audit.
`--harmonic-tfrecord-dir` only redirects the **compressor-training** iterator.
Compression is auto-read from the TFRecord manifest. Recorded in the run
`.meta.json` as `harmonic_tfrecord_dir` (deliberately **not** in the cache
fingerprint, so it never invalidates existing compressed `.npz` caches).

If `--harmonic-tfrecord-dir` is **not** set, the original `.npz` loader runs
unchanged (the default).

## How it works (data path)

```
TFRecord shards (/nas)
  └─ tf.data: interleave → shuffle(4096) → parse(decode_raw+reshape)
              → channel slice → channel scale → in-graph LR/UD flip (train)
              → batch(drop_remainder) → repeat → prefetch(AUTOTUNE)
  └─ DLPack zero-copy (tf CPU tensor → JAX) → device_put(GPU)   ← maps
  └─ theta: .numpy() + H0/100 on host                           ← theta (tiny)
  └─ yield {"maps": <jax device array f32>, "theta": <np f64>}
```

- **Converter** (`build_harmonic_tfrecord.py`): one shard per `.npz`, raw
  `float32`/`float64` `tobytes()` (bit-exact), zero-mean asserted per shard,
  idempotent, writes a `tfrecord_manifest.json` with a content hash. CPU-only
  (lazy per-worker TF import, `CUDA_VISIBLE_DEVICES=""`).
- **Reader** (`build_harmonic_tfrecord_iterator` in `npe_cnn_nbody_tomo.py`):
  the pipeline above. Flip is **in-graph** (cheap, overlaps compute). `maps` is
  handed to JAX via **DLPack** (no slow `.numpy()` copy) and placed on the GPU.

## Performance

| path | it/s | note |
|---|---|---|
| `.npz` loader (original) | ~2.4 | GIL-bound zlib decode; GPU ~90% idle (starved) |
| TFRecord, numpy flip | ~7 | flip on main thread was the cap (175 ms/batch) |
| **TFRecord, in-graph flip + DLPack→GPU** | **~17** | current; **~7.4×** over `.npz` |
| tf.data pipeline ceiling | ~24 | what the data path alone can produce |

A 80k-step production compressor: ~9.3 h → **~1.3 h**.

**Host threading (important — was a perf regression 2026-05-29).** Both the
tf.data decode/batch *and* the per-step host work in compressor training are
host-CPU-bound. This node's login shell exports `OMP_NUM_THREADS=1` (+ MKL/
OpenBLAS/NumExpr), which throttles them to a single thread → **~1 it/s with the
GPU starved at ~0% util** (looks like a broken DLPack handoff but isn't — maps
are verified on GPU). The script now self-corrects, in two places, governed by
one knob `CNN_TF_THREADS` (default 32, capped by available CPUs):
- sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` **before `import numpy`** (the
  numpy/MKL host work in the training step), and
- sets `tf.config.threading.set_intra/inter_op_parallelism_threads` after
  `import tensorflow` (the tf.data reader).

So you **no longer need any thread env prefix** — `--train-compressor` runs at
**~15 it/s under the default `OMP=1` shell** (validated; bit-exact gate intact).
Caveat: under heavy multi-tenant load (load avg ~40+) these threads still
compete for CPU and throughput drops, so also run when the node has free cores.

## Correctness

Bit-identical to the `.npz` path — proven, not assumed:
- §4.1 equivalence: raw patches, theta (post H0/100), slice, scale all **max abs
  diff 0.0**.
- §4.4 contract: the production reader's in-graph slice/scale/H0 also **0.0**.

Run the tests (CPU-only except throughput/smoke):

```bash
conda run -n jaxili python scripts/sbi/tests/test_tfrecord_equivalence.py   # §4.1 gate
conda run -n jaxili python scripts/sbi/tests/test_tfrecord_split.py         # §4.2
conda run -n jaxili python scripts/sbi/tests/test_tfrecord_epoch.py         # §4.3
conda run -n jaxili python scripts/sbi/tests/test_tfrecord_contract.py      # §4.4
```

## Decisions & caveats (read before extending)

- **In-graph flip** deviates from spec §6 ("flip in numpy"). Approved
  2026-05-28: flip is stochastic augmentation (§1.7), so the exact per-patch
  sequence need not match the `.npz` path — only its distribution does. This is
  what unblocked throughput (numpy flip was a 175 ms/batch ceiling).
- **Global shuffle is kept** (4096 buffer → ~8 cosmologies per 128-batch, vs the
  `.npz` pool's ~3.7). VMIM is batch-composition-sensitive, so **this is NOT a
  transparent drop-in**: the TFRecord-trained compressor has different training
  dynamics than the `.npz` one. To compare fairly, **re-run BOTH the L1 and CNN
  arms** under the TFRecord path. Approved 2026-05-28 (global shuffle is the
  better compressor; accept the behavioral change).
- **DLPack must target the accelerator explicitly.** `jax.device_put(arr,
  jax.devices()[0])` — without the device argument the host-imported buffer
  stays on `CpuDevice` and the whole training step silently runs on CPU (~10×
  slowdown observed 2026-05-28). Validate optimizations with the integrated
  training run + GPU util, never an isolated transfer micro-benchmark (an
  isolated benchmark hid this bug).
- The shared per-step NaN guard now dispatches via `_array_has_nan`
  (`jnp.isnan` for device arrays, `np.isnan` for numpy); the `.npz`/TFDS/paired
  paths are byte-for-byte unchanged.

## Known remaining headroom (deferred — may revisit)

Current ~17 it/s vs the ~24 it/s pipeline ceiling. The gap is **per-step host
synchronization barriers** in the shared `train_compressor_vmim` loop that
prevent JAX from pipelining data-load with compute:

1. `store_loss.append(float(b_loss))` reads the loss to host **every step**
   (biggest barrier) — accumulate on-device, read at log cadence instead.
2. The per-step `bool(jnp.isnan(maps).any())` guard — drop or make periodic for
   the harmonic route (cache is validated NaN-free at build time).
3. Optionally add device-prefetch double-buffering (`flax.jax_utils.
   prefetch_to_device` or a 2-deep background thread).

Expected: ~17 → ~24 it/s (~40%), run ~1.3 h → ~0.9 h. **Tradeoff:** these edits
are in the *shared* training loop (affects `.npz`/TFDS/paired-BNT too), so larger
blast radius and more revalidation for a modest gain — hence deferred. Use
`scripts/sbi/tests/profile_tfrecord_pipeline.py` to re-measure when picking this
up; the loss-readout change is the low-risk, high-value first step.

## Files

- `build_harmonic_tfrecord.py` — `.npz` → TFRecord converter.
- `npe_cnn_nbody_tomo.py` — `build_harmonic_tfrecord_iterator`,
  `_list_harmonic_tfrecord_shards`, `_resolve_harmonic_tfrecord_compression`,
  `_array_has_nan`, the `--harmonic-tfrecord-dir` flag + factory branch.
- `tests/test_tfrecord_*.py` — equivalence / split / epoch / contract /
  throughput.
- `tests/profile_tfrecord_pipeline.py` — stage-by-stage host-cost profiler.
