# Efficiency audit — GPU/CPU utilization across the flat-sky pipeline (2026-06-10)

Second audit round (3 parallel read-only agents), commissioned by Andreas with the **updated
resource envelope: GPU pool 0/1/2 (max-out allowed, tenant-checked; GPU 3 never), ≤50 CPU
workers**. Companion to `PIPELINE_AUDIT_2026-06-10.md` (whose §d found the un-jitted sampling
loop). Nothing changed yet — this is the findings + fix ladder for sign-off.

## Headline

Per-GPU slot occupancy in today's multiseed run was ~94%, but **effective GPU utilization was
~42%** — every sweep slot drives an A100 at ~45% SM / 608 MiB. The lever is **jobs-per-GPU, not
tail-trimming**: the repo's own June-9 logs show 2–3 sweeps co-resident per GPU with **0–4%
per-job penalty** (field-validated packing), yet every orchestrator hardcodes 1 job/GPU.
Per-process, the three standing wastes are the un-jitted sampling loop (known), the jaxili NDE
**training** loop (host-dispatch-bound: ~0.8 ms per jitted call that contains ~tens of µs of
A100 compute), and per-step host syncs in the compressor loop. Compressor training itself is
the healthy one: 88–93% util at ~100 it/s.

## Corrections to things we believed

1. **"On-device cross build costs ~20%" (felt fiber A2 note) is wrong.** Run-level medians over
   all production logs: none 91.7 it/s, product 95.4, both 98.4, conv 84.1; the in-flight s42/s43
   compressors ran 95–105 it/s. The old 93-vs-75 pairing was ambient-load coincidence; by
   construction the tf.data CPU work is identical across ops (loader reads only the 4 raw autos;
   cross build + whiten is one jitted GPU op). Conv's ~8% deficit is confounded (both ⊃ conv
   channels and was fastest). Fiber to be corrected.
2. **Compressor throughput is overhead-limited, not compute-limited** — but comfortably fast:
   plain-CNN train step ≈ 48 GFLOPs at batch 128 ⇒ ~2.5 ms fp32-peak floor vs 10.75 ms measured
   (~23% of non-TC fp32 peak). The gap is per-step host syncs (below), not tf.data (which has
   proper AUTOTUNE map + prefetch + bounded interleave).

## Fix ladder

### Tier 0 — usable tonight, no code changes [FREE]
- **Packing via duplicated GPU ids** on the thread-per-entry campaign runners (they spawn one
  worker per `--gpus` list entry, no dedup): `--gpus 1,1,2,2 --xla-mem-fraction-by-gpu
  1:0.45,2:0.45`. Works today on `run_cnn_noise_curriculum_campaign.py`,
  `run_tarp_dumps_campaign.py`, `run_cross_only_campaign.py`.
- Sweep packing baseline from measurement: **3 jobs/GPU** (memory ceiling irrelevant at 608 MiB;
  CPU ≈ 1.5 cores/job ⇒ 9 concurrent ≈ 14 of 50 cores).

### Tier 1 — small diffs, do with batch B [SAFE]
1. **Multi-slot greedy schedulers**: in every slots-dict runner, `slots = {(g,k): None ...}` with
   a per-phase packing table `{compressor: 1, fidsumm: 2, sweep: 3, tarp: 2, lc2st: 2, repr: 2}`
   and per-job `XLA_PYTHON_CLIENT_MEM_FRACTION = headroom/slots`. (~10 lines per script, or one
   shared helper.)
2. **Tenant probe at every launch**: lift `_probe_gpu_mem_fraction` from
   `run_cross_only_campaign.py:505-535`, add a skip rule (foreign memory > ~8 GB ⇒ skip/derate)
   and GPU-3 exclusion. Evidence this is mandatory: a foreign tenant colonized GPU 1 within
   **7 minutes** of it going idle today.
3. **Env preamble in every orchestrator `launch()`**: `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8`
   (4 when packing >4 jobs), `TF_NUM_INTRAOP_THREADS=8`, `TF_NUM_INTEROP_THREADS=2`,
   `CNN_CPU_THREADS=8`. Today's safety comes from login-shell `=1` exports; the live sweep
   process has **523 threads** (idle TF pools), and the L1 entry + `train_jaxili_from_compressed`
   set no caps at all (precedent for the failure mode: the documented 1237-thread lock-thrash
   incident). Also copy the `npe_cnn_nbody_tomo.py:47-91` thread-budget block into
   `npe_l1norm_cross_jaxili_nbody_tomo.py` and `train_jaxili_from_compressed.py`.
4. **Footguns**: `run_flatsky_cnn_repr_corners.py` defaults `--gpus 1,2,3` (forbidden GPU 3) —
   change default; `run_multiseed_compressor_check.py` / `run_flatsky_cnn_fiducial_summaries.py`
   build commands with `np.load(meta)` inside `launch()` → an earlier-phase failure kills the
   whole driver instead of skipping the chain (adopt `run_flatsky_l1_matrix.py:120-125`
   SKIP-on-dep-fail).
5. `overnight_cnn_pipeline.sh` serializes lc2st and repr-corners, which are mutually independent
   (~2.7 h recoverable on diagnostics nights when packed across 3 GPUs).

### Tier 2 — shared scheduler [needs sign-off; with/after BNT campaign build]
`scripts/sbi/gpu_scheduler.py` (~120 lines, assembled from existing pieces): probed pool
(0/1/2, tenant-skip, GPU-3 never) + per-class slot capacities + dep-graph jobs with
SKIP-on-dep-fail + launch-time command construction + one log line per launch recording
(gpu, mem-fraction, foreign GB, load1). The multiseed-style drivers become 12-job dep graphs;
pipelining falls out of deps for free (worth ~5–8 min alone, real wins come combined with
packing). Estimated campaign-shape gains (arithmetic from measured per-job times; compressor
packing is the conditional): today's 4-chain shape 4 h 42 m → **~2.5–2.7 h**; a 12-job sweep
campaign ~11 h → **~4.2 h** on 9 slots.

### Per-process fixes [gain = MEASURE; benchmarks specified]
1. **Jit the sampling loop — MEASURED + ADOPTED (2026-06-10).** `bench_sample_jit.py` on GPU 2:
   eager 183 ms/obs → jit **1.05 ms/obs (174×)**, 2.7 s one-time compile; vmap adds nothing
   worth the complexity. Bit-identity fails at the TF32 kernel level (max|Δ| 3.4e-3 on samples;
   same keys/u-draws), so the adoption gate was the full-arm rerun: `validate_jit_sweep.py`
   re-derived none_s42's complete 9000-obs pooled median in **49 s** (vs ~4100–4800 s eager) with
   median FoM3 −0.39%, σ's within ±0.2% — an order of magnitude inside seed scatter
   (`multiseed/population_sweep/none_s42/jit_validation.json`). Wired into
   `population_sweep_flatsky.py` as the default; `--sample-eager` reproduces the legacy path.
   Sweeps are now NDE-training-dominated (~30 min/arm instead of ~100).
2. **jaxili NDE training**: the epoch loop is Python-per-batch over a dataset that jax-dataloader
   forces back to *host numpy* (`jax_dataloader/loaders/jax.py:18-66` — `asnumpy()`, host gather,
   fresh H2D per batch, `num_workers` ignored). Measured ~0.8 ms/jitted call vs tens-of-µs
   compute; dim-3200 costs ≈ dim-10 (the loop never sees the FLOPs). Fix: device-resident train
   split (≤2.9 GB even for L1-both) + `lax.scan` epoch, reproducing the per-epoch permutation
   (statistically identical; plausibly bit-identical). FREE riders: drop the dead
   `jnp.asarray(theta_tr/x_tr)` in `train_jaxili_from_compressed.py:165-166`;
   `donate_argnums` on train_step. RECIPE option (sign-off): `check_val_every_epoch=2`
   (val pass = 22% of per-epoch calls).
3. **Compressor loop syncs**: `float(b_loss)` + `jnp.isnan(b_loss)` every step block async
   dispatch; the flat_local input NaN guard syncs the transform before the update dispatches.
   Move loss materialization + NaN checks to the 100-step log cadence (bit-identical params
   trajectory; only failure-detection latency changes); keep numpy-route NaN checks off the
   critical path. Hypothesis ≥30% it/s — benchmark before claiming.
4. **L1 build pass** is loader-bound at ~174 maps/s vs 486/s compute ceiling — the perm filter
   decodes 1.13 M examples to keep 323 k (3.5× decode waste). SAFE: raise interleave for finite
   passes (changes row order only; pairs stay matched), or materialize the perm-filtered autos
   subset once (~33 GB, fits the RAID). RECIPE (don't): fp64→fp32 wavelets — pointless while
   loader-bound and breaks the GATE-A bit-match oracle.
5. **BNT campaign needs no dataset rebuild** (on-device 4×4 channel mix + L1 noise re-freeze);
   if anyone proposes a cache rebuild instead, that re-triggers ~1.2 h SHT + 7.7 h serial TFDS
   writer — push back.

## Measurement plans (per `feedback_benchmark_dont_assume`; ≥3 reps, record load + co-tenants)
- Sampling jit/vmap: `bench_sample_jit.py` (bit-identity + FoM3 gates built in).
- Sweep 3/GPU + 4/GPU same-day controlled: solo vs packed on one GPU, `sampled N/9000` slopes;
  accept largest N with aggregate ≥ 0.9 × N × solo.
- Compressor 2/GPU: solo vs 2 concurrent (frac 0.45), accept if each ≥ 0.85× solo it/s.
- Cross-class (sweep+compressor co-resident): compressor it/s ≥ 0.9× solo to accept.
- Compressor sync-removal: 2k-step A/B, same loss curve to fp tolerance.
- NDE scan-epoch: one CNN + one L1 arm, wall time + best_eval + FoM3 parity (bit-compare epoch-1
  params if order preserved).
- lc2st/repr 2/GPU (CPU-heavier): watch host load specifically.

## CPU-side verdict (third agent)
Healthy: builders are budget-exact (Pool(50) × 1-thread pinned workers), TFDS on local XFS RAID
(not NFS; 4.8 TB free), `build_fiducial_summaries_cnn.py` fine as-is (180-patch batches, ~2
min/arm), no multiprocessing retrofit warranted anywhere; `np.savez` vs compressed choices are
right. The only CPU action items are the thread-cap preamble (above) and noting a TFDS rebuild
is wall-clock-bound (~8 h serial writer), not CPU-bound.
