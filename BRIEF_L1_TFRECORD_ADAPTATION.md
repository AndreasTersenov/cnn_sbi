# BRIEF — adapt the L1 cross script to TFRecord + datavector reuse

**For**: the session that just built the CNN harmonic-TFRecord path (you're warm — you wrote
`build_harmonic_tfrecord_iterator`, the equivalence tests, and `HARMONIC_TFRECORD_README.md`).
**Created**: 2026-05-28. This is **Part 1** of a 2-way split (Part 2 = run continuation, a separate session).

## Your task (one sentence)

Do for the **L1 cross script** (`scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py`) what you did for the CNN
script — make it read the harmonic data from TFRecord — **plus** add a datavector disk-cache so the L1
training datavector is computed once per arm and reused across all seeds×perms. Result: L1 auto+cross
goes from ~8 h/arm to ~1 h/arm, with **bit-identical** L1 datavectors.

## Where the full spec lives (don't duplicate — read these)

The detailed, exact spec is in `HANDOFF_L1_TFRECORD_AND_RELAUNCH.md`. Your scope is **Parts §1a, §3, §4, §4b, §5**:
- **§3** — the L1 harmonic read sites; the clean strategy (add an optional `tfrecord_dir` to the single
  `iter_harmonic_examples`, so all three callers — dataset + both calibrations — speed up at once).
- **§4** — the exact contract the TFRecord branch must preserve.
- **§4b** — the datavector-reuse restructure (the dominant speedup).
- **§5** — the validation gate (bit-exact L1 datavector + flip-invariance + cache round-trip).
- **§1a** — why L1 must NOT copy the CNN's global shuffle.

Do **not** do §6/§7 (running the arms, CNN-side work) — that's Part 2.

## The four L1-specific gotchas (different from your CNN reader — do not autopilot)

1. **NUMPY, not JAX device arrays.** L1 wavelet stats run in PyTorch (`torch.from_numpy`). Reuse your
   parse logic (`tf.io.decode_raw` + reshape, the Example schema) but **drop the DLPack/device_put and the
   in-graph flip** — those were JAX-compressor-specific. Yield numpy.
2. **channel_scale MULTIPLIES** in L1 (`patches = patches * scale`, ~line 397) — your CNN reader **divides**
   (`maps_np / channel_scale`). Do not cross them; match `_load_harmonic_file`.
3. **theta stays RAW** (H0=68.5). The L1 pipeline divides H0/100 downstream (~line 777), not in the reader.
   Your CNN reader applied /100 at read; the L1 reader must NOT.
4. **Per-shard 48-patch blocks in sorted-stem order. NO global shuffle.** L1 has no VMIM compressor and
   jaxili shuffles the NDE internally, so sorted-order shards keep the L1 result a results-neutral speed-only
   change. (Your CNN global shuffle was a deliberate, CNN-only behavioral change — see §1a.)

## The datavector reuse (the part that's new vs your CNN work) — §4b — ⚠️ CONDITIONAL

**Correction (2026-05-28):** the earlier claim that the L1 statistic is *flip-invariant* is WRONG — you
measured flip changes the L1 datavector ~10% (cross-channels + finite-patch boundary effects break the
equivariance). So dedup-via-`flip=False` is NOT automatically lossless. Dedup viability now hinges on **the
flip-FoM3 experiment you're already running** (flip=True vs flip=False L1 auto+cross, same seed, compare
FoM3/σ):

- **If flip=False ≈ flip=True in FoM3** (flip aug doesn't help L1): set the train datavector `flip=False`
  (currently `flip=True` ~line 2499) — now exactly deterministic → **enable the dedup cache**. The datavector
  then depends only on (split, channel config); perm = obs only, seed only drives NDE init/split. The 9
  runs/arm self-deduplicate.
- **If flip=True is meaningfully better** (flip aug helps L1): **keep `flip=True` and forgo the dedup.** Do NOT
  trade accuracy for speed, and do NOT give L1 weaker augmentation than the CNN (which keeps flip=True). The
  TFRecord reader port alone still delivers the ~3.3× format speedup per run.

Dedup cache (only if the experiment greenlights flip=False): in `compute_l1_dataset_from_harmonic_cache`, key on
(regime, split, channel_slice, channel_scale, l1_nbins, n_scales, SNR ranges, subtract_coarse_mean,
l1_implementation) — **not** seed/perm/format. Load-if-exists; else compute + save atomically.

**Do the reader port now regardless** (it's independent of the flip result). Gate the dedup on the experiment.

## Add the flag

`--harmonic-tfrecord-dir` (+ auto-read compression from the manifest), mirroring the CNN script. Keep
`--full-sphere-cross-cache` required (obs via `load_observed_from_harmonic_cache` stays on `.npz`; it's one
fast read and bit-identical). Record `harmonic_tfrecord_dir` in the run `.meta.json`.

## Gate (don't hand back until all green)

1. `scripts/sbi/tests/test_l1_tfrecord_equivalence.py` (you write it, per §5): L1 datavector `.npz` vs TFRecord
   **max abs diff 0.0** (or documented ≤1e-5), for slice=None and slice(0,4), with channel_scale.
2. Flip-FoM3 experiment resolved (the one you're running). If it greenlights flip=False → dedup cache added + round-trip bit-exact. If not → flip=True kept, dedup skipped (documented). Either way the reader port is done + bit-exact.
3. Your existing CNN suite still passes (`test_tfrecord_equivalence.py` etc.) — you didn't disturb shared code.
4. A 50-epoch both-paths L1 smoke runs; the TFRecord-path log shows
   `cross_noise_model = channel_empirical_global` + the channel_scale table (NOT a fallback warning).

## Stay in your lane

- Edit **only** `npe_l1norm_cross_jaxili_nbody_tomo.py` (+ the new test). The split-slicing fix is already in
  it (`_parse_harmonic_split_slice`, ~line 300) — don't redo it.
- Don't touch the shared NaN guard (`_array_has_nan`, already yours) or the CNN script.
- GPU policy (2026-05-28): **GPU 1 max, GPU 0 ≤45%** (other tenant), GPUs 2/3 off-limits.

## Hand back

When the gate is green, append to the fiber `.felt/definitive-l1-vs-cnn-2026-05/...` Loop Status:
"L1 TFRecord + datavector-reuse landed + validated (equivalence max-abs-diff <X>, flip-invariance <Y>,
speedup <Z>× measured)." Then the Part-2 session runs the L1 auto+cross arms per
`HANDOFF_L1_TFRECORD_AND_RELAUNCH.md` §6.
