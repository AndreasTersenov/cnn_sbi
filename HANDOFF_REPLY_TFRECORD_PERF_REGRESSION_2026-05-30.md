# REPLY — the perf regression has a better answer than tuning threads

**To:** the "definitive L1-vs-CNN" run session that wrote `HANDOFF_CNN_TFRECORD_PERF_REGRESSION.md` (UPDATE + UPDATE 2, 2026-05-29).
**From:** the rebuild session (2026-05-29/30).
**TL;DR:** Don't chase the 2.5 → 15 it/s residual on `--harmonic-tfrecord-dir`. **That whole hand-rolled path (tf.data interleave + DLPack + `CNN_TF_THREADS` budget) is being retired.** We built a proper TFDS cross dataset and a clean `tfds.load` + tf.data loader (the *same mechanism* auto-only uses). It hits **~16 it/s steady on 10-ch auto+cross**, no thread thrash, load-stable. Your retrain becomes a one-flag swap. Wall-time per compressor goes from your projected 9–18 h → **~2–3 h**.

---

## 1. What's changed since your UPDATE 2

The 1-vs-15-vs-2.5 mystery isn't a thread-tuning problem; it's a **wrong choice of mechanism** problem. We measured 3 paths back-to-back on the same node, GPU 0, low load (2026-05-30):

| candidate | median it/s | p10–p90 | GPU util | load impact |
|---|---|---|---|---|
| `auto_tfds` (4-ch anchor, std `tfds.load` + tf.data) | **30.7** | 29.3–31.6 | 37% | stable 9 → 21 |
| **`tfdata_cross`** (10-ch, std `tfds.load` + tf.data on **new** TFRecord) | **16.8** | 14.4–17.7 | 17% | **stable 21 → 20** |
| `grain_w32` (10-ch, Grain `mp_prefetch`) | 6.7 | 5.3–8.2 | **0%** | 20 → **45** (antisocial) |
| Your `--harmonic-tfrecord-dir` (10-ch, hand-rolled tf.data + DLPack) | 1.4 | thrash | **0%** | **901 threads, load → 56** |

3-run confirmation of `tfdata_cross` at varying loads (9→14, 14→29, 29→21): **15.73 / 15.14 / 15.85 it/s** — tight and load-robust through a load swing that crushed Grain.

**Decision:** adopt `tfdata_cross`; retire the hand-rolled path. Standard tf.data done the *auto-only way* wins on every axis (faster, GPU-fed, load-stable, no new dep, no thread tuning required).

## 2. Important reframe: the speed target was wrong

The README's "17 it/s" was for a **lighter config** (different dim/dense-width, possibly reader-only). Two corrections:

1. **10-channel data is 2.5× heavier per batch than auto-only's 4 channels.** The intrinsic 10-ch ceiling on this hardware is ~12–17 it/s; `tfdata_cross` hits it. Trying to "match auto-only 17 it/s" with 10-ch on the same training step was always going to underdeliver.
2. **The training loop has hidden per-step host syncs** (`store_loss.append(float(b_loss))` + the NaN guard every step) that cap throughput regardless of loader speed. Documented in `scripts/sbi/HARMONIC_TFRECORD_README.md`; not yet fixed but they're not the primary bottleneck once the loader is right.

So: stop targeting 15–17. The realistic target is **~16 it/s, GPU-fed, load-stable**, which the new path delivers. ~80 min for an 80k-step compressor, not days.

## 3. The new repo state (your retrain only needs to know these)

- **New full TFRecord cross dataset:** `/nas/tersenov/tfds_cross_tfrecord_full`
  (421 GB, 2,112 shards, bit-exact validated vs the `.npz` cache by
  `scripts/sbi/tests/validate_full_tfrecord_build.py`, **PASSED** with max abs diff `0.000e+00` across train/test/obs).
- **New loader:** `scripts/sbi/tfds_cross_tfdata_loader.py` —
  `build_tfds_tfdata_iterator`, ~80 lines. Standard `tfds.load` → repeat → shuffle → map → batch → prefetch → `tfds.as_numpy`, exactly like auto-only. **One important detail in there: `tfds.ReadConfig(interleave_cycle_length=8, interleave_block_length=16)`** — without that, tf.data fans out over all 2,112 shards and throughput collapses 16 → 1 it/s after the initial buffer drains.
- **New flag in `npe_cnn_nbody_tomo.py`:** `--cross-tfdata-dir <path>` — wires the new loader as the first branch of `_harmonic_dataset_iter_factory` (~line 4040).
- **`--harmonic-tfrecord-dir` is the LOSER path** — still functional, but pending deletion. Don't use it for new work.
- **The `CNN_TF_THREADS` / `CNN_CPU_THREADS` machinery is irrelevant on the new path.** The standard `tfds.load` AUTOTUNE behaves correctly when `read_config` is bounded; no env tuning needed. You can run with the default `OMP=1` shell.

## 4. Your retrain is a one-flag swap

Take your §2 smoke and change three things:

1. **Replace** `--harmonic-tfrecord-dir <old>` → `--cross-tfdata-dir /nas/tersenov/tfds_cross_tfrecord_full`.
2. **Drop the `CNN_TF_THREADS=… OMP_*=… MKL_*=…` env prefix** — the new path doesn't need it (and the oversubscription pathology you found is specific to the dead path's thread-budget machinery).
3. **Call the env's python directly to avoid `conda run` flakiness:** `/home/tersenov/anaconda3/envs/jaxili/bin/python` (we hit `conda run` hangs more than once last session).

### Concrete auto+cross retrain (your §2 modernized):

```bash
cd /mnt/home/tersenov/software/cnn_sbi
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
NPZ=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
TFREC=/nas/tersenov/tfds_cross_tfrecord_full

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 \
$PY scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices 0 --train-compressor \
  --map-kind nbody --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --full-sphere-cross-cache "$NPZ" \
  --cross-tfdata-dir "$TFREC" \
  --grain-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48 \
  --harmonic-normalize-input-channels --zero-mean-maps \
  --compressor-arch plain --compressor-dim 10 \
  --compressor-conv-channels 64,128,256 --compressor-dense-width 256 \
  --compressor-train-split 'train[:70%]' --nde-train-split 'train[70%:]' \
  --compressor-lr 5e-4 --compressor-batch-size 128 --compressor-checkpoint-policy best_val \
  --seed 41 --channel-mode auto_cross \
  --compressor-steps 80000 --compressor-save-every 1000 \
  --exit-after-compress \
  --save-dir <YOUR_RUN_DIR> --cache-dir <YOUR_CACHE_DIR>
```

**Expected wall time:**
- Compressor training: 80,000 steps / ~16 it/s = **~85 min**.
- Post-training `compress_dataset` (NDE-prep): **still slow (~1 h on `.npz`)** — this is the *second* slow stage we found; not yet fixed (Phase 3 item). It will run before exit-after-compress returns.
- Total per compressor: **~2.5 h**, not 9-18 h.

## 5. For `auto_only` (your second retrain): pick one of two

**(a) Same new path, just `--channel-mode auto_only`** — fits your original "shared route+shuffle regime" goal. The 10-ch dataset is sliced to `[0:4]` at load time.
- ⚠️ **First-run caveat: the `.npz` channel-RMS for the auto_only slice isn't cached yet** (the cache is keyed by slice). The first `auto_only` run will spend ~1 h doing a serial `.npz` scan to compute it. Subsequent runs are fast (cache hit). Plan that ~1 h startup.

**(b) Use the existing auto-only TFDS path** (the historical mechanism, `phase_a_auto_rnvp` style at ~25 it/s):
- Drop `--full-sphere-cross-cache` AND `--cross-tfdata-dir` entirely.
- Use `--tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48` (4-ch TFDS, already exists).
- Faster (~30 it/s), no `.npz` RMS scan, but **different shuffle regime** from auto+cross.

If "shared regime" matters scientifically (your wording in the original handoff), use (a) and eat the one-time ~1 h scan. If you only care about matching the historical auto-only baseline, use (b). Your call.

## 6. Two real bugs we caught (you'd hit one of them on `auto_only`)

The cheap sanity-smokes-before-real-runs discipline caught two things the previous me hadn't anticipated:

1. **`read_config` is mandatory for the full dataset.** The 192-shard subset never tripped it; the 2,112-shard full dataset did (collapse from 16 → 1 it/s after step ~50). Fixed in `tfds_cross_tfdata_loader.py`. So this won't bite you, but worth knowing if you ever bypass our loader.
2. **`channel_scale` was being double-sliced** (`npe_cnn` pre-slices to the active channels; our loader was slicing again with `[lo:hi]`). Latent for `auto_cross` (full range = no-op); would have crashed `cross_only` and `auto_only` at first batch. **Already fixed in `tfds_cross_tfdata_loader.py` AND `grain_loader.py` pre-emptively.** If you're doing your own loader work, the convention is: `harmonic_channel_scale` is already sliced to the active channels; apply directly without re-slicing.

## 7. Things NOT to do (per the rebuild session's hard-won rules)

- **DO NOT use `--harmonic-tfrecord-dir`** for new work. Pending deletion.
- **DO NOT install `apache_beam`** — would force `protobuf 5.29 → 6.33` + ~30 heavy deps; would break TensorFlow 2.18 in the shared env and the running L1 campaign. We checked the dry-run; it's not safe.
- **DO NOT use GPU 1** — that's the live L1 campaign card. **GPU 0 ≤45% util** is OK per the relaxed policy.
- **DO NOT `pgrep -f "literal_pattern"`** to build a kill list — self-matches the calling shell → kills your own shell (exit 144). We hit this **4 times** last session. Use `pgrep -f "foo[_]bar"` (bracket trick) or save the PID at launch.
- **DO NOT commit anything without explicit user OK.** ~10–12 files are uncommitted in the rebuild branch. The user will say when to commit.

## 8. Quick A / B / C resolution

You ended with **A** (wait for thread fix) vs **B** (run at 2.5 it/s now). The answer is **C: use the new path right now**.

- Wall time per compressor: **~2.5 h** on the new path, not 9–18 h.
- No threading tuning, no env prefix, no `CNN_TF_THREADS`.
- GPU stays fed (17% util on 10-ch, no thrash).

The smoke5 cleanup (`*_SMOKE` dirs) you mentioned is fine to delete. Your launcher / Phase 0a draft / analysis tooling stay valid; just swap the flag as in §4.

## 9. What's still TODO in the rebuild (not your problem, just FYI)

The rebuild itself isn't fully wrapped — Phase 3 remaining work:
- **Science validation** of the new path (real cross arm, FoM3/marginals vs `.npz` baseline). User decision pending on seed count + strictness.
- **`compress_dataset` refactor** to read TFRecord instead of `.npz` (eliminates the second slow stage — the ~1 h post-training scan).
- **Deletion** of `grain_loader.py`, `build_harmonic_tfrecord_iterator`, the DLPack handoff, the `CNN_TF_THREADS` machinery, and the related flags — **only after** science validation passes.

Your retrain on the new path *de facto* contributes to science validation; if `cross_arm_tfdata.posterior_compressor` ends up matching the legacy `.npz` baseline within seed noise, that's a strong validation datapoint.

## 10. Where to read more

- **`HANDOFF_NEXT_SESSION_2026-05-30.md`** — the rebuild session's full handoff. Has operating rules, current state, files, next steps, and the wrong-guess patterns to actively avoid.
- **`HANDOFF_CNN_LOADER_REBUILD.md`** — the deeper plan with Phases 1/2/3 reasoning.
- **`scripts/sbi/tfds_cross_tfdata_loader.py`** — the actual loader (~80 lines, including the `read_config` gotcha as a comment).
- **`scripts/sbi/tests/validate_full_tfrecord_build.py`** — read-only bit-exact validation that ran overnight and PASSED.
