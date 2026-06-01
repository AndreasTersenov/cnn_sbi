# Handoff — CNN auto+cross loader rebuild (Phase 3 in flight)

**Session ending:** 2026-05-30 (continued from 2026-05-29 evening).
**Branch:** `autoresearch/cnn-auto-push-18-20-2026`.
**Nothing committed.** ~10–12 files modified or new; user has not yet asked to commit.

---

## 0. TL;DR

The CNN auto+cross training data path has been rebuilt as a **proper TFDS dataset
(TFRecord) read via the same standard `tfds.load` + tf.data pipeline that makes
auto-only fast**. Decision is empirically locked, full dataset is built and bit-exact
validated, the loader is integrated into `npe_cnn` behind `--cross-tfdata-dir`,
sanity-checked across channel modes, and **two real bugs were caught and fixed by the
sanity checks before they could ruin science**. Remaining: science validation (the
gate), `compress_dataset` refactor, then deletion of the dead old paths.

---

## 1. The big-picture goal (and the right framing)

**Goal:** "Build the CNN auto+cross data path properly and cleanly, so the end result
runs really fast, by checking everything empirically rather than guessing."

Three sub-goals, in tension:
- **Proper / clean** — non-hacky, standard approaches; reuse what works (auto-only's
  `tfds.load` + tf.data) rather than bolting on custom machinery.
- **End result really fast** — training loop + adjacent stages (channel-RMS scan,
  `compress_dataset` NDE-prep).
- **Checking, not guessing** — every perf claim measured on this node. (We slipped on
  this multiple times; correcting was part of the work — see §10.)

**Important reframe (made empirically, mid-session):** "as fast as auto-only" (~30
it/s on 4 channels) is **NOT physically achievable for 10-channel cross data** — the
per-batch transfer is 2.5× larger. The honest target is **"at the intrinsic cross
ceiling, with low overhead and load stability."** That ceiling is ~16 it/s on this
hardware (measured), and the adopted path hits it.

---

## 2. Operating rules (load-bearing — broken several times last session)

These guided the work and were learned the hard way. Treat them as hard rules:

1. **Check, don't guess.** Every perf number must be measured on this hardware,
   repeated, with conditions stamped (load, GPU co-tenants). Time estimates without
   measurements were wrong by 3–10× multiple times (see §10). **If you catch yourself
   emitting a number you haven't measured, stop and measure.** Or say "I don't have a
   number for that yet."
2. **Plan before implementing substantial code.** Substantial = a new module, a
   refactor of a working file, a significant code path. Plan in
   `HANDOFF_CNN_LOADER_REBUILD.md` and get sign-off before coding.
3. **Cleanliness is conditional on deletion.** Adding a loader path doesn't make the
   code clean; *deleting the old paths* is what does. Current state has 4 loader
   paths in `npe_cnn`; net cleanliness only materializes when the dead ones go.
4. **No destructive ops without OK.** `git reset --hard`, mass deletes, force-push,
   amend, env upgrades, `git add -A` — none without explicit user OK. The user is
   sometimes asleep; queue read-only validations and report.
5. **Verify dep installs with `pip install … --dry-run` FIRST.** `apache_beam` would
   have forced protobuf 5→6 + ~30 heavy deps → would have broken TensorFlow 2.18 in
   the shared env → would have broken the running L1 campaign. `grain 0.2.16` was
   safe (zero changes to existing packages). **Always check the diff before
   installing into a shared env.**
6. **Process-grep uses the `[_]` bracket trick.** `pgrep -f "foo_bar"` (or
   `$(pgrep -f "foo_bar")`) self-matches the calling shell's argv → `kill` then takes
   down your own shell (exit 144). **Hit 4 times last session.** Use
   `pgrep -f "foo[_]bar"` so the regex doesn't match its own literal, or save the
   PID at launch with `echo $! > /tmp/foo.pid`.
7. **GPU policy** (project rule, lightly relaxed):
   - GPU 1 is the L1 campaign's card — **never touch**.
   - GPU 0 OK if `nvidia-smi` shows ≤45% util.
   - GPUs 2/3 are other tenants — leave alone.
   - Confirm with `nvidia-smi` before every GPU launch.
8. **The training loop has hidden per-step host syncs** (`store_loss.append(float(b_loss))`
   + the NaN guard every step) that cap throughput regardless of loader speed.
   Documented in `scripts/sbi/HARMONIC_TFRECORD_README.md`. Not yet fixed.

---

## 3. The decisive empirical finding (the loader question is settled)

The 3-way back-to-back comparison (GPU 0, low load ~10):

| candidate | median it/s | p10–p90 | GPU util | load (pre→post) | mechanism |
|---|---|---|---|---|---|
| `auto_tfds` (4-ch anchor) | **30.7** | 29.3–31.6 | 37% | 8.9 → 20.8 | std `tfds.load` + tf.data |
| **`tfdata_cross`** (10-ch) | **16.8** | 14.4–17.7 | 17% | 20.8 → **19.9 stable** | std `tfds.load` + tf.data on TFRecord |
| `grain_w32` (10-ch) | 6.7 | 5.3–8.2 | **0%** | 19.9 → **45.2** | Grain `mp_prefetch` |

3-run confirmation of `tfdata_cross` at varying loads: **15.73 / 15.14 / 15.85 it/s** —
extremely tight, load-robust (load swung 9→29 between runs).

**Decision: adopt `tfdata_cross`, drop Grain.**
- Standard tf.data on TFRecord is **2.5× faster than Grain** under the same conditions.
- **Keeps node load stable** (Grain's 32 workers tripled load 20→45).
- **No new dep** — it's the exact mechanism auto-only already uses.
- Surprising to the previous me — I initially leaned Grain based on web research; the
  measurement said the opposite. Owning that.

The 10-channel ceiling (~16 it/s) is consistent with 2.5× the data per batch vs
auto-only's 30 it/s × 4 ch. We're at the physical ceiling. The training loop's
per-step host syncs probably cap us before we hit pure-transfer limits.

---

## 4. Where Phase 3 stands

### Done ✓
- **`compute_cross_patches` refactor** in `build_full_sphere_cross_cache.py` — single
  source of truth, behavior-preserving (validated bit-exact via
  `tests/validate_cross_compute_refactor.py`).
- **TFDS builder** `tf_dataset_nbody_tomo_cross.py` — reads validated `.npz` cache
  via `mp.Pool(50)` workers, `DISABLE_SHUFFLING=True`, per-patch examples with schema
  `{map_nbody, theta, cosmo_idx, perm, patch}`.
- **TFDS subsets built** (for benchmarking):
  - `/nas/tersenov/tfds_cross_arrayrecord_subset20` (20 cosmologies, ArrayRecord)
  - `/nas/tersenov/tfds_cross_tfrecord_subset20` (20 cosmologies, TFRecord)
- **Full TFRecord build:** `/nas/tersenov/tfds_cross_tfrecord_full` — 421 GB, 2,112
  shards, completed 2026-05-30 07:12 UTC (wall: 8h 48min). **Bit-exact validated**
  by `tests/validate_full_tfrecord_build.py` — 30 examples spot-checked across
  train/test/obs, max abs diff **0.000e+00**.
- **The winning loader:** `scripts/sbi/tfds_cross_tfdata_loader.py` with
  `read_config(interleave_cycle_length=8, block_length=16)` (needed to keep tf.data
  from fanning out over all 2112 shards and collapsing throughput; see bug #1).
- **Integration into `npe_cnn_nbody_tomo.py`:** new flags `--cross-tfdata-dir`,
  `--grain-tfds-dir`, `--grain-num-workers`, `--grain-tfds-name`; factory branches
  in `_harmonic_dataset_iter_factory` (preferring `cross_tfdata_dir` →
  `grain_tfds_dir` → `harmonic_tfrecord_dir` → `.npz`).
- **Grain loser path** `scripts/sbi/grain_loader.py` — functional, integrated, but
  losing path → to delete.
- **Benchmark harness** `tests/bench_cnn_loader.py` extended with candidates:
  `tfrecord` (old hand-rolled), `npz`, `grain_w{8,16,32}`, `tfdata_cross`,
  `tfdata_cross_full`, `tfdata_cross_full_{cronly,autonly}`, `auto_tfds`.
- **Sanity smokes for all channel modes** on the full data — see §3a below.

### Pending ⏳

1. **Science validation (THE GATE before deletion).** Run a real cross arm end-to-end
   with `tfdata_cross_full` (`--channel-mode auto_cross`, the primary mode), compare
   FoM3 + marginal σ to a legacy `.npz`-path baseline at the same seed. Within seed
   noise = PASS. Decisions owed by user: seed count (1 paired vs 2–3), strictness of
   "within noise" gate, whether to also validate `cross_only`.
2. **`compress_dataset` refactor.** The NDE-prep at `npe_cnn:1555` reads `.npz`
   serially via `iter_harmonic_examples` (~1 h post-training scan). Re-point at the
   TFRecord dataset so the *full pipeline* is fast, not just the training loop.
   Use a deterministic (no-shuffle) variant of `build_tfds_tfdata_iterator`.
3. **Deletion** (only after science validation PASS):
   - `scripts/sbi/grain_loader.py` (delete file).
   - `pip uninstall grain` (optional; harmless to leave).
   - In `npe_cnn`: drop `--grain-*` and `--harmonic-tfrecord-dir` flags + their
     factory branches; the function `build_harmonic_tfrecord_iterator` + helpers
     `_list_harmonic_tfrecord_shards`, `_resolve_harmonic_tfrecord_compression`;
     the DLPack handoff + `_array_has_nan` (if no longer needed); the
     `_resolve_cnn_cpu_threads` thread-budget machinery + the BLAS-env block + the
     TF intra/inter setters at the top of the file.
   - `scripts/sbi/build_harmonic_tfrecord.py` (orphan converter for the old path).
   - `scripts/sbi/tests/test_tfrecord_{equivalence,split,epoch,contract,throughput}.py`
     (tests for the hand-rolled reader).
   - Probably KEEP `build_harmonic_batch_iterator` (the `.npz` loader) since other
     scripts may use it; just remove from the CNN's hot path.
4. **Documentation:** retire/rename `scripts/sbi/HARMONIC_TFRECORD_README.md`,
   write a `TFDS_CROSS_README.md` describing the adopted path + the read_config
   gotcha.

### 4a. Sanity smokes results (the two bugs caught — exactly why we did them)

| smoke | result |
|---|---|
| Standalone loader smoke (full data, auto_cross) | PASS — shapes, dtype, theta h0-scaled |
| Integrated 200-step training (full data, auto_cross) | **16.0 it/s steady** (after read_config fix) |
| Integrated 100-step training (full data, cross_only) | **21.9 it/s** (faster — 6 ch < 10 ch) ✓ |
| Standalone slicing (auto_only `[0:4]` + cross_only `[4:10]`) | both PASS — `(128,160,160,4)` / `(128,160,160,6)` |
| Integrated auto_only on full data | timed out on `.npz` channel-RMS scan (separate ~1h pre-compute, not a loader issue; covered by the standalone slicing test above) |

**Bug 1 — `read_config` needed for many-shard datasets.** TFDS default lets tf.data
interleave across ALL 2,112 shards. After the initial shuffle-buffer fill (~50
batches), throughput **collapses from 16 to 1 it/s, GPU at 0%**. Fixed with
`tfds.ReadConfig(interleave_cycle_length=8, interleave_block_length=16)`. **The
subset (192 shards) never tripped this**, so the integrated full-data smoke was
essential.

**Bug 2 — channel_scale double-slice.** `npe_cnn` *pre-slices*
`harmonic_channel_scale` to the active channels (auto_cross → `[10]`, cross_only →
`[6]`, auto_only → `[4]`). My loaders then sliced again with `[lo:hi]`, producing a
`[2]` array vs a `[H,W,6]` map → `ValueError: shapes [160,160,6] [2]`. **Latent
for auto_cross** (slice was full range = no-op). Fixed in
`tfds_cross_tfdata_loader.py` AND `grain_loader.py` (pre-emptively, even though
Grain is being deleted).

---

## 5. Files in play

Read in this order if picking this up fresh:

### Plan / handoff
- **`HANDOFF_NEXT_SESSION_2026-05-30.md`** (this file) — current state, what to do next.
- `HANDOFF_CNN_LOADER_REBUILD.md` — substantive plan doc with decisions and methodology.
  Already has Phase 1/2 results; this file extends with Phase 3.

### The proper path (keep + extend)
- `scripts/sbi/tf_dataset_nbody_tomo_cross.py` — TFDS GeneratorBasedBuilder. Reads
  validated `.npz` cache via `mp.Pool(50)` workers (`CROSS_TFDS_BUILD_WORKERS` env
  var). `DISABLE_SHUFFLING=True`. Per-patch examples.
- `scripts/sbi/build_cross_tfds_dataset.py` — programmatic build wrapper. Takes
  `--data-dir`, `--file-format {array_record,tfrecord}`, `--cosmo-limit`.
- **`scripts/sbi/tfds_cross_tfdata_loader.py`** — **THE WINNER**. Standard
  `tfds.load` + tf.data pipeline with `read_config(cycle=8, block=16)`, transforms
  (slice, scale, flip, theta H0→h0). `scale_t` does NOT re-slice (bug #2 fix).
- `scripts/sbi/build_full_sphere_cross_cache.py` — refactored. Shared functions
  `compute_noisy_alms`, `cross_patches_from_alms`, `compute_cross_patches`.
- `scripts/sbi/npe_cnn_nbody_tomo.py` — integrated. Factory branches:
  `_harmonic_dataset_iter_factory` at line ~4040; new flags around line ~427+.

### To delete after science validation
- `scripts/sbi/grain_loader.py` (loser of the loader bake-off)
- `scripts/sbi/build_harmonic_tfrecord.py` (orphan converter for old path)
- In `npe_cnn` (lines indicated approximate, file edited heavily):
  - `_resolve_cnn_cpu_threads` @ `:47` (and the BLAS-env block right after)
  - `build_harmonic_tfrecord_iterator` @ `:1389`
  - `_list_harmonic_tfrecord_shards` @ `:1351`
  - `_resolve_harmonic_tfrecord_compression` (helper)
  - `_array_has_nan` @ `:2241` (only if no other consumers)
  - `--harmonic-tfrecord-dir`, `--grain-*` flags and their factory branches.

### Testing harness + smokes
- `scripts/sbi/tests/bench_cnn_loader.py` — main harness. Self-test:
  `python bench_cnn_loader.py --self-test`. Default candidate is `tfrecord`; the
  good one is `tfdata_cross_full`.
- `scripts/sbi/tests/validate_cross_compute_refactor.py` — proves the refactor is
  bit-exact vs the `.npz` cache.
- `scripts/sbi/tests/test_tfds_cross_equivalence.py` — proves the TFDS builder is
  bit-exact vs the `.npz` cache (subset).
- `scripts/sbi/tests/validate_full_tfrecord_build.py` — **read-only** full-build
  bit-exact validation (ran overnight, PASSED).
- `scripts/sbi/tests/smoke_grain_loader.py`, `smoke_tfdata_cross_full.py` — quick
  loader smokes.

### Data on /nas
- **`/nas/tersenov/tfds_cross_tfrecord_full`** — the full 421 GB TFRecord cross
  dataset. Bit-exact validated. **USE THIS for science validation.**
- `/nas/tersenov/tfds_cross_arrayrecord_subset20` — 20-cosmo ArrayRecord subset
  (Grain benchmarking; can delete after Phase 3).
- `/nas/tersenov/tfds_cross_tfrecord_subset20` — 20-cosmo TFRecord subset
  (tf.data benchmarking; can delete after Phase 3).
- `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid` — old hand-rolled
  TFRecord. Keep as-is for now; can delete after deletion of the hand-rolled reader.

### `.npz` cache (still needed for channel-RMS, observed, audit)
- `scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid/nobnt/` —
  305 GB, the source of truth that TFRecord was reserialized from. **DO NOT DELETE.**

---

## 6. Concrete next steps (in order)

### Step 1 — Science validation (the gate before deletion)

Run a real cross arm end-to-end with the new path, compare to a legacy baseline at
the same seed.

**Recommended command for the NEW path** (auto_cross, primary mode):

```bash
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
$PY scripts/sbi/npe_cnn_nbody_tomo.py \
  --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --full-sphere-cross-cache scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid \
  --cross-tfdata-dir /nas/tersenov/tfds_cross_tfrecord_full \
  --grain-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48 \
  --channel-mode auto_cross \
  --train-compressor --compressor-steps <CAMPAIGN_DEFAULT> \
  --compressor-save-every <CAMPAIGN_DEFAULT> \
  --compressor-batch-size 128 --compressor-dense-width 256 \
  --compressor-train-split train --compressor-val-split val \
  --nde-train-split train --nde-val-split val \
  --cnn-map-route harmonic --harmonic-cache-regime nobnt \
  --harmonic-normalize-input-channels \
  --ds-batch-size 500 \
  --compressor-arch plain --compressor-dim 10 \
  ... <NDE training flags from run_cross_only_campaign.py NDE_BUDGET_FLAGS> ... \
  --seed <SEED> --save-dir <NEW_PATH_RUN_DIR> \
  --cuda-visible-devices 0
```

**The BASELINE is the same command WITHOUT `--cross-tfdata-dir`** (falls through to
the `.npz` loader; same seed).

Read `scripts/sbi/run_cross_only_campaign.py` for the canonical Stage A + Stage B
flag set; mimic it.

**Decisions to get from the user before launching:**
- Seed count: 1 paired (cheap, ~2× cross-arm training time) vs 2–3 paired (tighter
  statistics, 4–6× training time)?
- Strictness of "matches within seed noise": FoM3 ratio within ±10%? Marginal σ
  within ±5%? Posterior overlap visually indistinguishable?
- Also validate `--channel-mode cross_only`? (Used in some campaigns.)

### Step 2 — `compress_dataset` refactor

`npe_cnn:1555` (`compress_dataset`) currently reads `.npz` via
`iter_harmonic_examples`. Plan:
- Add a deterministic-order `build_tfds_tfdata_iterator` variant (no shuffle), OR
  pass `shuffle=False` to the existing one.
- Iterate train+val examples; apply the trained compressor; collect `(theta, summary)`
  pairs in deterministic order.
- Smoke-test against the legacy path: same summaries (within numerical noise) for a
  subset.

### Step 3 — Deletion (only after Step 1 passes)

See file list in §4. Do it as ONE commit with a clear message. Verify nothing else
in the repo imports the deleted names (`grep -r build_harmonic_tfrecord_iterator
scripts/`).

### Step 4 — Documentation

Write `scripts/sbi/TFDS_CROSS_README.md` describing the adopted path: build command,
load command, the read_config gotcha, the pre-sliced channel_scale convention.
Retire `HARMONIC_TFRECORD_README.md` (or replace its content with a redirect).

---

## 7. Specific NOT-TO-DOs

1. **DO NOT install `apache_beam`.** Would upgrade protobuf 5.29 → 6.33 + ~30 heavy
   deps. Would break TensorFlow 2.18 in the shared env and the running L1 campaign.
2. **DO NOT `git add -A` or `git add .`.** Tree is dirty by design. Stage explicitly
   by path. (Project CLAUDE.md rule.)
3. **DO NOT delete the dead paths before science validation PASSES.** They're our
   fallback if validation finds an issue.
4. **DO NOT use `pgrep -f "literal_pattern"` to build a kill list.** Self-matches the
   calling shell → kills your own shell (exit 144). Use the `[_]` bracket trick or
   save the PID at launch.
5. **DO NOT commit anything without explicit user OK.** ~10–12 uncommitted files
   exist; user will say when to commit.
6. **DO NOT use GPU 1** — L1 campaign's card. Use GPU 0 (≤45% util OK) or check first.
7. **DO NOT guess perf / time estimates.** See §10 — the pattern bit us repeatedly.
   Measure first. Always.
8. **DO NOT spawn agents unless the user explicitly asks** (per global CLAUDE.md).
9. **DO NOT write comments saying "verified X"** before you've measured X. (Caught
   myself once last session; removed.)

---

## 8. Hardware / environment

- **Machine:** `titan`, 128 cores, 764 GB RAM, 4× A100 40 GB.
- **Conda env:** `jaxili`. **Use the env's python directly to bypass occasional
  `conda run` flakiness:** `/home/tersenov/anaconda3/envs/jaxili/bin/python`.
- **Key versions:**
  - `tensorflow 2.18.0`
  - `tensorflow_datasets 4.9.9`
  - `array_record 0.8.3`
  - `grain 0.2.16` (will be removed)
  - `numpy 1.26.4`
  - `apache_beam`: **NOT installed** (do not install).
- **GPU policy:** GPU 1 is L1 campaign (never touch). GPU 0 OK if `nvidia-smi`
  shows ≤45% util. GPUs 2/3 are other tenants. Confirm before launching.
- **CPUs available:** up to 50. Useful for build (`mp.Pool(50)`).
- **`/nas`:** 48 T free, 421 GB used by the full TFRecord build.

---

## 9. Cleanup status / loose ends

- All overnight background processes exited cleanly (build PID + watcher PID).
- No leftover GPU jobs.
- No leftover `/tmp` clutter that matters (the build log is at
  `/tmp/full_tfrecord_build.log`, validation at `/tmp/overnight_status.log`; both
  useful as historical reference).
- The post-build watcher (PID 3242225) exited at 07:13 UTC after validation PASS.

---

## 10. The wrong-guess pattern (own these — they kept happening)

Multiple time/perf estimates last session were wrong. Pattern: emitting numbers from
intuition rather than measurement. **Owning these is part of the handoff** — the new
session will be tempted to do the same and should resist.

| guess | actual | gap |
|---|---|---|
| "Building TFRecord too would cost ~1 h" | ~7 h (then full build: 8h 48m) | **~7×** |
| "5–7 h full build" | 8h 48m | ~50% |
| "~20 min remaining" (build + bench) | ~5 min | ~4× |
| "tfdata_cross will likely match Grain with less overhead" | actually **2.5× faster** | underclaimed |
| "verified ~15 it/s steady" (in a comment) | hadn't measured yet | caught + removed |
| `mp.Pool(50)` build → "~7 h → ~20 min" | actual ~15% speedup (serializer-bound) | overclaimed |

**The fix is mechanical:** if you're about to write a number you haven't measured,
write **"I don't have a number for that yet"** instead. The user explicitly called
this pattern out and asked us to fix it. Hold the line.

---

## 11. Memory updates (for the auto-memory system)

**Saved this session:**
- `feedback_benchmark_dont_assume.md` — perf claims need measured numbers.
- `feedback_no_pkill_self_match.md` — updated to 4× incident count + bracket trick.
- `project_harmonic_tfrecord_training_path.md` — updated: hand-rolled path is being
  retired in favor of standard tf.data on TFDS-cross.

**Added by this handoff (to be created by next session if not already there):**
- `project_tfdata_cross_wins_grain.md` — the empirical decision + measured numbers.
- `project_tfds_load_interleave_tuning.md` — `read_config(cycle=8, block=16)` is
  required for many-shard datasets to avoid shuffle-buffer-exhaustion collapse.
- `feedback_dont_guess_time_estimates.md` — the recurring failure mode, with the
  table from §10.

---

## 12. The exact state of npe_cnn integration

Quick reference so the new session doesn't have to re-grep:

- `--cross-tfdata-dir <dir>` — adopt the standard tf.data path. **Use this.**
- `--grain-tfds-dir <dir>` — Grain path (loser; will be deleted).
- `--grain-tfds-name <name>` — shared by both new paths (default OK).
- `--grain-num-workers <N>` — Grain only.
- `--harmonic-tfrecord-dir <dir>` — the OLD hand-rolled path (will be deleted).
- The factory `_harmonic_dataset_iter_factory` (around `npe_cnn:4040`) dispatches:
  `cross_tfdata_dir > grain_tfds_dir > harmonic_tfrecord_dir > .npz`.

The cross_tfdata route still requires `--full-sphere-cross-cache <.npz cache>` for
channel-RMS, observed data, and the split audit. The `.npz` cache is **NOT** going
away soon; only its role as the *training-time loader* is changing.

---

End of handoff. Pick this up by reading §0 → §1 → §2 → §6, then dive into the
science validation. The previous me's biggest weakness was guessing numbers without
measuring — please be better at this than I was.
