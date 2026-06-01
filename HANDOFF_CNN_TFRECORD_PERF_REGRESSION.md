> 🛑 **SUPERSEDED (2026-05-30) — DO NOT ACT ON THIS DOC.** The perf regression is RESOLVED.
> Root cause was **storage** (the dataset on `/nas` FUSE mergerfs ~100 MB/s), not DLPack /
> threads / mem-fraction. Fix: read from the **local xfs copy**
> `/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/` → ~15–19 it/s.
> The old `--harmonic-tfrecord-dir` path here has been **deleted**; a normalization bug in
> the new loader was **fixed** (be on HEAD ≥ `676d407`). **Read
> `HANDOFF_PERF_REGRESSION_RESOLVED_2026-05-30.md` instead.** Everything below is historical.

# HANDOFF — CNN TFRecord fast-path is running at ~0% GPU util (perf regression)

**For**: the session that built the harmonic-TFRecord CNN acceleration (`d3e8cc8`,
`scripts/sbi/HARMONIC_TFRECORD_README.md` / `..._IMPLEMENTATION_SPEC.md`).
**From**: the "definitive L1-vs-CNN" run session, 2026-05-29.
**TL;DR**: I tried to retrain both CNN compressors on the harmonic TFRecord path. The
wiring is correct and the data is read, but **steady-state throughput is ~1.1 it/s with
GPU 0 at ~0% utilization** — i.e. the compressor is effectively training on the host, the
GPU is idle. That is ~15× below the README's documented **~17 it/s**.

> ⚠️ **READ THE `UPDATE (2026-05-29)` SECTION AT THE BOTTOM FIRST.** My original prime
> suspect below (§6.1, DLPack/jax-0.5.0) was **ruled out** by the acceleration session — the
> DLPack handoff is fine and jax 0.5.0 is not the cause. Their threading fix (`526b12e`) solved
> the **reader**. But the REMAINING problem stands: with the fix active + light load, **full
> compressor training is still ~1.1 it/s with GPU ~idle** — the wall is now per-step
> training-loop overhead, not the reader. §§5–6 below are kept for the diagnostic record; the
> current, correct status is in the UPDATE.

---

## 0. Environment (where this was measured)
- `jax 0.5.0`, `jaxlib 0.5.0`, `tensorflow 2.19.0`, conda env `jaxili`.
- Node: 128 physical CPUs, but **the shell exports `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`** (so `nproc`
  reports 1 — `nproc` honors `OMP_NUM_THREADS`). cgroup `cpu.max = max` (no quota); shell
  affinity = all 0–127.
- Heavy multi-tenant load at test time: `uptime` load avg ~40–53; GPUs 1/2/3 occupied by
  other users (`bonjean`, `titan`, `alahiry`), **GPU 0 free** — all runs below were on GPU 0.
- `/nas` read bandwidth measured fine: **1.5–1.8 GB/s** for a TFRecord shard
  (`dd` of a 49 MB shard); local `.npz` 4.6 GB/s. So `/nas` is NOT the bottleneck.

## 1. What I was trying to do
Retrain BOTH CNN compressors on the **harmonic TFRecord** so they share one route+shuffle
regime (auto+cross via `--channel-mode auto_cross`, auto-only via `--channel-mode auto_only`,
4-ch slice). Launcher: `scripts/sbi/run_cnn_retrain_tfrecord.sh`. Reconstructed-consistent
config (logs+fiber+script defaults), RealNVP companion (script default; no
`--vmim-companion-backend` flag exists). `--exit-after-compress` → train compressor + write
`cnn_{train,val,obs}.npz`.

## 2. Exact repro (smoke that exhibits the problem)
```bash
cd /mnt/home/tersenov/software/cnn_sbi
NPZ=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
TFREC=/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32 \
TF_NUM_INTRAOP_THREADS=32 TF_NUM_INTEROP_THREADS=8 \
conda run --no-capture-output -n jaxili python scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices 0 --train-compressor \
  --map-kind nbody --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --full-sphere-cross-cache "$NPZ" --harmonic-tfrecord-dir "$TFREC" \
  --harmonic-normalize-input-channels --zero-mean-maps \
  --compressor-arch plain --compressor-dim 10 \
  --compressor-conv-channels 64,128,256 --compressor-dense-width 256 \
  --compressor-train-split 'train[:70%]' --nde-train-split 'train[70%:]' \
  --compressor-lr 5e-4 --compressor-batch-size 128 --compressor-checkpoint-policy best_val \
  --seed 41 --channel-mode auto_cross --compressor-steps 500 --compressor-save-every 250 \
  --exit-after-compress --save-dir /tmp/smk --cache-dir /tmp/smk_cache
```

## 3. Symptom (the regression)
- First step ~39 s (XLA compile + shuffle-buffer fill — expected).
- **Steady-state (step ~234–241): ~1.10–1.22 it/s.**
- **GPU 0 utilization (`nvidia-smi dmon -i 0 -s u`): 0% almost continuously** (one stray 8%
  sample), while JAX holds ~7.5 GB on GPU 0. → GPU is idle; the step time is host-bound.
- Loss IS decreasing (`Step 250 | train -6.25 | test -8.21`), so it trains correctly — just
  ~15× too slow. README target for this exact arm (auto+cross plain): **~17 it/s**.

## 4. What I CONFIRMED is fine (so you can skip these)
- **Wiring is correct.** `npe_cnn_nbody_tomo.py:4012` selects `build_harmonic_tfrecord_iterator`
  whenever `--harmonic-tfrecord-dir` is set (no silent `.npz` fallback). `:3689` `auto_only`→
  `slice(0,4)`; `:3694` hard-errors if `auto_only`/`cross_only` used off the harmonic route;
  `:3724` RMS norm computed post-slice (length matches the slice). Both channel-modes read the
  TFRecord.
- TFRecord present + valid: `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid` (6293/2800/91
  shards, manifest, NONE compression). Reader reports `train[:70%]: 4405/6293 shards`.
- `--harmonic-normalize-input-channels` works (RMS ratio 41896× printed).
- `/nas` bandwidth fine (§0). NOT an I/O-bandwidth problem.

## 5. What I RULED OUT as the cause
- **`/nas` bandwidth** — 1.5–1.8 GB/s, ample (§0).
- **CPU thread-limit env vars** — the shell defaults are `*_NUM_THREADS=1`. I re-ran with
  `OMP/MKL/OPENBLAS/NUMEXPR=32` + `TF_NUM_INTRAOP=32 / INTEROP=8`. Throughput moved only
  **0.9 → ~1.1 it/s** and GPU util stayed ~0%. So the single-thread default is NOT the wall
  (or TF/tf.data parallelism isn't honoring those env vars — see §6.2).

## 6. Prime suspects (for you to confirm with the profiler)
### 6.1 DLPack→GPU handoff under jax 0.5.0 (most likely)
The reader (`npe_cnn_nbody_tomo.py:1476-1479`) does:
```python
maps_dev = jax.device_put(
    jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(maps_tf)),  # line 1477
    target_device,   # jax.devices()[0]
)
```
Under **jax 0.5.0** this emits:
> `DeprecationWarning: Calling from_dlpack with a DLPack tensor is deprecated. The argument
> to from_dlpack should be an array from another framework that implements the __dlpack__
> protocol.`
Hypothesis: in jax 0.5.0 the deprecated capsule path (`to_dlpack` → `from_dlpack(capsule)`)
no longer yields a proper device-placed / zero-copy array, so `device_put` operates on a
host buffer and the model step runs host-side (matches 0% GPU util). The README explicitly
warns that a host-backed maps array makes "the whole training step run on CPU (~10× slowdown)";
this looks like that failure mode resurfacing via the API change.
- **Candidate fix**: `jax.dlpack.from_dlpack(maps_tf)` directly (tf EagerTensors implement
  `__dlpack__` in tf 2.19), drop `tf.experimental.dlpack.to_dlpack`. **Then verify** the result
  actually lands on GPU (`maps_dev.devices()` / `.sharding`) and that GPU util rises — silencing
  the warning is NOT sufficient evidence; measure util + it/s.
- **Re-run the bit-exact gate after any change**: `tests/test_tfrecord_equivalence.py`
  (must stay max-abs-diff 0.0) — the equivalence is the whole point; don't trade it for speed.

### 6.2 tf.data parallelism may be capped
Even with `TF_NUM_INTRAOP/INTEROP` set, tf.data `num_parallel_calls=AUTOTUNE` may be sizing off
a throttled CPU count (TF can read cgroup/`nproc`). Worth checking the actual tf.data thread
usage and whether explicit `options.threading.private_threadpool_size` helps. (Secondary —
0% GPU util points more at 6.1 than at data-prep starvation, since even a starved-but-on-GPU
step would show util spikes.)

### 6.3 Per-step host syncs (README "known headroom")
The README already flags `store_loss.append(float(b_loss))` and the per-step
`bool(jnp.isnan(maps).any())` as 17→24 it/s headroom. These are NOT the 1.1 it/s cause but are
relevant once 6.1 is fixed. `tests/profile_tfrecord_pipeline.py` is the intended tool to
re-measure the stage breakdown — please run it first; it will name the slow stage.

## 7. Secondary bug found + already fixed in the launcher (FYI)
`--total-steps` does NOT control compressor training length. The compressor uses
**`--compressor-steps`** (`npe_cnn_nbody_tomo.py:487`, default **150000**). With `--total-steps`
the smoke trained toward 150000 instead of the intended count. The original campaign
compressors used 80000 (`logs/phase_a_*_rnvp.log`: "steps=80000"). The retrain launcher now
passes `--compressor-steps`.

## 8. Artifacts / where to look
- Launcher: `scripts/sbi/run_cnn_retrain_tfrecord.sh` (smoke + train modes).
- Smoke logs: `scripts/sbi/results/exploratory/definitive_comparison/logs/cnn_retrain_autocross_tfrec_rnvp_SMOKE.log`.
- Original compressor logs (for config + the 80k steps + 2.14 vs 25.6 it/s baselines):
  `.../logs/phase_a_{auto,autocross}_rnvp.log`. NOTE: original `auto_rnvp` was **TFDS route**
  (25.6 it/s, never slow); `autocross_rnvp` was **harmonic `.npz`** (2.14 it/s — the slow one
  TFRecord was meant to fix). Neither used `--harmonic-tfrecord-dir`.
- Profiler: `scripts/sbi/tests/profile_tfrecord_pipeline.py`; equivalence gate:
  `tests/test_tfrecord_equivalence.py`.

## 9. Bottom line for the acceleration session
The acceleration is correctly plumbed into both CNN pipelines, but on `jax 0.5.0` the
DLPack→GPU handoff appears to no longer place maps on the accelerator, so training is
host-bound at ~1.1 it/s / ~0% GPU util. Please: (1) run `profile_tfrecord_pipeline.py` to
confirm the slow stage; (2) update the `from_dlpack` call to the non-deprecated `__dlpack__`
form and verify maps land on GPU + util rises; (3) re-run `test_tfrecord_equivalence.py` to
keep bit-exactness. Once it's back to ~17 it/s, the retrain command in §2 (with
`--compressor-steps 80000 --compressor-save-every 1000`, and the `auto_only` variant) is
ready to go on GPU 0.

---

## UPDATE (2026-05-29, after pulling the fix `526b12e`)

Pulled `526b12e` (it was already in HEAD on the shared branch). The TF-threading fix
is confirmed active and your reader diagnosis was right — but **full compressor training
is still ~1.1 it/s with GPU ~idle**, which is a *different* bottleneck than the reader you
validated. Evidence (my env: fix active, node load ~9 (light), GPU 0, no env prefix):

- `[cnn-tf-threading] intra=32 inter=8 (avail_cpus=128)` prints — fix is in effect.
- **Profiler** (`profile_tfrecord_pipeline.py`, on GPU 0) stage rates:
  `1_base 17.4 | 2_+shuffle 26.1 | 3_+flip 25.9 | 4_+numpy(reader) 9.7 | 5_+H2D 8.4 it/s`.
  So the pipeline can deliver ~8–26 it/s.
- **Full `--train-compressor` smoke (auto_cross, 10ch): steady ~1.1 it/s**, GPU 0 ~0% util
  (only brief ~8% `dmon` blips). So ~8× slower than the reader can feed.

**Conclusion**: the wall is now the **per-step training-loop overhead**, not the reader. The
GPU is near-idle while each step takes ~900 ms, and the reader can supply a batch in ~100 ms,
so ~800 ms/step is host-side per-step cost — consistent with the README's own "known headroom"
(`store_loss.append(float(b_loss))` + per-step `jnp.isnan(maps).any()` forcing a device→host
sync every step, blocking prefetch/overlap). In full training this appears to *dominate*, not
just cap 17→24 — my 1.1 it/s is even below the "~7 it/s" your profiler docstring references.

**Three discrepancies vs your validation (for you to consider — likely-confounding, check FIRST):**
0. ⚠️ **CONFIG MISMATCH (most likely the dominant confounder).** My full-training run used the
   **heavy reconstructed campaign config**, which is much heavier than the `npe_cnn` defaults:
   `--compressor-dim 10` (default 6), `--compressor-conv-channels 64,128,256` (default 32,64,128),
   `--compressor-dense-width 256` (default 64), plus `--harmonic-normalize-input-channels` and the
   **full VMIM step** (compressor + RealNVP companion, ~1.04 M params) via `--train-compressor`.
   If your "~17 it/s" was measured on a lighter/default config — or on a reader-only loop — then
   the 1.1-vs-17 gap is **partly (maybe mostly) config + the model step**, not a pure regression.
   **Please reproduce full training with the EXACT §2 command (heavy config) and report its it/s +
   GPU util**; tell us what config your 17 it/s used so it's apples-to-apples. (The reader-pipeline
   rates — profiler stages 1–3, 17–26 it/s — are config-independent and look fine; the open
   question is full-training throughput at THIS config.)
1. My reader stage-4 is **9.7 it/s**, not your **19.8**. Likely because I ran with **no OMP
   prefix** (shell `OMP_NUM_THREADS=1`), so the `.numpy()` host-copy + any numpy host ops are
   single-threaded. Your in-code fix sets *TF* threads but not OMP/MKL; the numpy host side may
   still want `OMP_NUM_THREADS>1`. (Production uses DLPack not `.numpy()`, so stage-4 isn't the
   exact production consumer, but the numpy-thread point likely still bites the per-step host work.)
2. You validated the **reader** (19.8 it/s), not full `--train-compressor` throughput. The
   1.1 it/s here is full training. Please measure full-training it/s + GPU util, not just the
   reader.

**Asks for the acceleration session:**
- Address the per-step host syncs in `train_compressor_vmim` (accumulate loss on-device, read at
  log cadence; make the NaN guard periodic for the harmonic route which is validated NaN-free;
  consider `flax.jax_utils.prefetch_to_device` double-buffering). The README lists these as
  headroom — but they appear to be the *primary* full-training cost, not a 40% tweak.
- Confirm whether OMP/MKL threads also need raising for the host-side per-step work (not just TF).
- Re-validate on **full training** (it/s + `nvidia-smi dmon` util), and keep
  `test_tfrecord_equivalence.py` at max-abs-diff 0.0 after any loop change.

Until full-training throughput is fixed, the retrain is impractical (~1.1 it/s ⇒ ~20 h per 80k
compressor). The §2 command + the `auto_only` 4-ch variant are ready to fire once it's resolved.

---

## UPDATE 2 (2026-05-29, after pulling `2cc57f9` — root cause of the 1-vs-15 gap)

Pulled `2cc57f9` (host-BLAS fix; in HEAD). On GPU 1 (free), load ~9.6, my exact §2 heavy
config, full `--train-compressor`, **NO env prefix** — diagnosed the remaining gap to a
**thread oversubscription** triggered by the default `CNN_TF_THREADS=32` on this 128-core node:

| setting | threads (NLWP) | proc state | GPU util | it/s |
|---|---|---|---|---|
| `CNN_TF_THREADS=32` (default) | **1237** | `Sl` (blocked/sleeping), ~44% of 1 CPU | **~0%** | ~1.0 |
| `CNN_TF_THREADS=8` | **37** | `Rl` (running), ~385% CPU | **88–93%** | ~2.5 (var 1–3) |

- jax is correctly on **CudaDevice(1)** in both cases (gpu backend, ~4.5 GB) — NOT a CPU fallback.
- At the default 32, the TF threadpool + OMP/MKL/OPENBLAS/NUMEXPR(32) + tf.data AUTOTUNE (sizing
  off 128 cores) **stack to ~1237 threads**; the process thrashes/blocks (Sl, GPU starved at 0%) →
  ~1 it/s. This is why your `526b12e`+`2cc57f9` give you 15 it/s but me ~1: **the default thread
  count oversubscribes in my environment.**
- `CNN_TF_THREADS=8` collapses that to 37 threads, unblocks the GPU (88–93% util), → ~2.5 it/s.

**Two things for you:**
1. The default `CNN_TF_THREADS=32` is unsafe on a busy 128-core node — consider capping tf.data
   parallelism explicitly (`options.threading.private_threadpool_size` / a bounded
   `num_parallel_calls`) rather than AUTOTUNE-off-128, and/or a lower default. The total thread
   count, not just the per-pool setting, is what matters.
2. **Residual gap unexplained**: even GPU-bound at ~90% util with threads=8, I get ~2.5 it/s (and
   variable 1–3), not your 15. Same A100-class card, same config. Possibly node contention, or a
   per-step difference (jit? step time?). What thread count + it/s + `ps -o nlwp` do you see, and
   was your 15 it/s under exclusive node use? A `CNN_TF_THREADS` value that auto-scales to *free*
   cores (not total) might be the robust fix.
