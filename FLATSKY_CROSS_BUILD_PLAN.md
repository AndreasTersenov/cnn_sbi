# Flat-sky cross-maps — build & analysis plan (for the next session)

**Goal:** replace the leaky full-sphere harmonic cross-maps with **patch-local flat-sky** cross-maps,
recompute the statistics, train L1 + CNN, and produce calibrated cosmological contours — to get the
*physically defensible* auto+cross result (the full-sphere one is partly unphysical; see
`CROSS_MAP_LEAKAGE_FINDING.md`). Design reasoning & validation already done:
`FLATSKY_CROSS_REDESIGN_NOTES.md` (read it first — §7–14 are load-bearing).

This plan is deliberately conservative and gate-driven. **Do not skip the validation gates.** The
history of this project is full of subtle, confounded conclusions (FoM3 fragility, noise-model
amplitude bugs, route confounds, lag/registration traps); the gates exist to stop us repeating them.

---

## 0. The big simplification — NO sim/dataset rebuild

The flat-sky cross-maps are a **deterministic function of the patch's own 4 auto maps**, so they can
be computed **on-the-fly** from auto channels 0–3 of the existing TFDS
`NbodyCosmogridDatasetTomo/grid_10deg_80px_nonoverlap180` (10 ch = 4 auto + 6 *harmonic* cross; we
discard channels 4–9 and recompute them flat-sky). **No PKDGRAV/SHT reprocessing.** This also means
the **auto-only baseline uses the identical auto channels**, so auto vs auto+cross is a clean,
confound-free comparison. The fiducial obs cache (`full_sphere_cache_fiducial_10deg`, 200 perms) auto
channels are likewise reused for the observed-map path.

---

## 1. Construction recipe (patch-local; both operators)

Inputs: the 4 auto maps κ₁…κ₄ per patch, **after `--zero-mean-maps`** (per-channel spatial demean) and
after optional BNT — exactly as the current pipeline produces them. Apodization window W = separable
cosine taper, **roll ≈ 12%** (`_apod_window_np`). For each of the 6 pairs (i<j):

- **Convolution cross map (Zürcher flat-sky analog):**
  `Cᵢⱼ = irfft2( rfft2(κᵢ·W) · rfft2(κⱼ·W) )` — **apodized circular** (standardized; do NOT use the
  zero-pad+crop variant — it only differs by a 39-px crop-offset shift + small edge wrap, notes §13).
  This is what `_compute_cross_maps_{np,tf}` already compute (just confirm the apodization width).
- **Product cross map:** `Pᵢⱼ = κᵢ · κⱼ` (pointwise; no FFT, strictly local). Decide whether to also
  multiply by W² for edge consistency — likely yes for the L1/SNR path so edges don't dominate.

Each arm appends the 4 raw auto channels + 6 cross channels → 10 channels. Keep the train and obs
implementations bit-identical (validate; this has bitten before). The FFT-product *math* in
`_compute_cross_maps_{np,tf}` is a correct *reference* — but see §4.5 (reuse audit) and §7
(performance): do NOT reuse its noise handling, and do NOT reuse its placement inside a CPU `tf.data`
map. Re-validate the math against `validate_flatsky_cross.py`; don't trust it blind.

> Operator rationale (do not relitigate; see notes §8–10, §12): convolution = multiply in Fourier
> (Zürcher-faithful, smooth/large-scale, carries cross-info in morphology); product = multiply in
> real space (its **mean = ξᵢⱼ**, scale-preserving, local). They are complementary, not redundant.

---

## 2. Noise / SNR normalization (THE fix for the old bug)

The old `--cross-maps` route fed all channels through the **shared auto pixel-σ** → wrong SNR for the
cross channels (`feedback_l1_cross_must_use_harmonic_route`; notes §5.6). The flat-sky cross-map noise
is **not** white auto-σ (validation V3: amplitude 0.33×σ, and for the product/coloured-noise case it
differs by scale). Two consistent, real-data-applicable options:

1. **Per-channel, per-scale SNR (Zürcher):** SNR = κ_filtered / σ(κ_filtered), where σ is estimated
   **per channel, per wavelet scale** from **noise realizations** (rotate galaxy ellipticities / add
   independent shape-noise draws, rebuild the cross map, take the filtered std). The arbitrary
   convolution/product amplitude then **cancels** automatically. This is the recommended, physically
   clean choice and is what real WL HOS analyses do.
2. **Minimal:** extend the existing `channel_empirical_global` per-channel-σ machinery (currently
   wired only to the harmonic route) to the flat-sky `--cross-maps` route. Faster to implement but
   per-channel only (not per-scale); acceptable as a first pass.

For the **CNN**, normalization is simpler: divide each channel by its **per-channel RMS** (as
`tfds_cross_tfdata_loader` already does via `channel_scale`); no wavelet/SNR. Add the flat-sky cross
augmentation + per-channel RMS.

---

## 3. Wavelet scales

Reuse the pipeline's starlet `n_scales` (same as the auto channels) for consistency. Expect the
**finest scales of the convolution cross to be near-empty** (it's smooth/large-scale) — that is fine
and physical; the cross info lives in coarser scales (validation: ⟨k⟩≈9 cycles/patch). Do **not**
re-introduce PCA (`--pca-components 0`, HARD RULE `feedback_never_pca_l1`).

---

## 4. Pipeline integration

- **L1:** the flat-sky route already exists (`npe_l1norm_cross_jaxili_nbody_tomo.py --cross-maps`).
  Work = (a) point it at the auto channels (slice 0–3 of the 10-ch TFDS, or a 4-ch auto TFDS), (b)
  add the **product** operator alongside the convolution (`--cross-op {conv,product,both}`), (c) wire
  the **per-channel(/per-scale) noise** (§2), (d) bit-match np obs path.
- **CNN:** `npe_cnn_nbody_tomo.py` has **no** flat-sky cross support — must ADD it (reuse
  `_compute_cross_maps_*` + per-channel RMS). Mirror the L1 `--cross-op` flag.
- **NDE:** jaxili MAF for L1; for CNN keep the campaign-exact compressor→NDE. **Example-disjoint
  compressor/NDE split BY PERM** (compressor perms 0–4, NDE 5–6), all cosmos in both (the prior
  campaign's decision).

---

## 4.5 Reuse audit — what to reuse / rewrite / DISCARD (be strict)

The old `--cross-maps` flat-sky route is the one we spent this session proving was bad
(`FLATSKY_CROSS_REDESIGN_NOTES.md` §5). Treat it as a **reference to critique, not a foundation**.

- **REUSE (after re-validating):** the apodization window `_apod_window_np` (but use ~12% roll, not
  8%); the FFT-product *formula* (one line) as the convolution operator — re-check it against
  `validate_flatsky_cross.py`, don't assume.
- **ADD (new):** the pointwise **product** operator; flat-sky support in `npe_cnn_nbody_tomo.py`
  (none exists).
- **REWRITE (the broken parts — do NOT carry over):**
  - the **noise/SNR** for cross channels — old route used a *shared auto-σ* (the documented bug);
    replace with per-channel(/per-scale) σ from noise realizations (§2). Discard the old
    `--cross-map-min/max-snr`, `--cross-map-auto-calibrate-snr`, `cross_snr_percentile` band-aids —
    they patched the symptom (histogram range), not the cause (the σ denominator).
  - the **computation placement** — old route runs the FFTs inside a CPU `tf.data` map
    (`_compute_cross_maps_tf`); that starves the GPU (§7). Move the augmentation **on-device,
    batched** (torch for L1, JAX for CNN), with `tf.data` delivering only the 4 auto channels.
- **VERIFY don't assume:** the old autos-appended-raw-while-cross-from-apodized is actually correct
  (we want raw autos, apodized cross inputs) — keep, but confirm. tf vs np (or torch vs np) bit-match
  is a GATE, not an afterthought.

## 5. Experiment matrix & decision rule

All arms share the identical auto channels; cross built patch-local.

| arm | L1 | CNN |
|---|---|---|
| auto-only | ✓ | ✓ |
| auto + **convolution**-cross | ✓ | ✓ |
| auto + **product**-cross | ✓ | ✓ |
| auto + **both** cross sets | only if conv & product are complementary | same |

- **PRIMARY METRIC (declare ONE):** median over typical patches of **σ(w₀) and 2D(Ωm,σ8) area**.
  FoM3 reported but **never headlined** (`feedback_fom3_fragile_use_2d_areas`).
- **Decision:** (i) does each patch-local cross set beat auto-only, and by how much? (ii) is the gain
  **calibrated** (TARP/SBC/L-C2ST) before believing it? (iii) are conv & product **complementary**
  (does "both" beat each alone)? (iv) how does the *honest* flat-sky gain compare to the inflated
  full-sphere number (expect: smaller — that is the point).
- **Expectation management:** the cross info is large-scale and a 10° patch samples it poorly →
  **modest gains are the physically correct outcome**, not a failure (notes §9.3).

---

## 6. Build / run sequence (gate-driven)

0. **Read** `FLATSKY_CROSS_REDESIGN_NOTES.md`, `CROSS_MAP_LEAKAGE_FINDING.md`, this plan, and the
   memory index. Recreate the env (`conda run -n jaxili`, GPU **1 only**).
1. **Implement** the augmentation **on-device, batched** (L1 `--cross-op` via torch.fft; CNN flat-sky
   via jax.fft; per-channel noise) — NOT in a CPU `tf.data` map (§7). Compile-check.
2. **GATE A — construction + throughput:** bit-match the two implementations; re-run
   `validate_flatsky_cross.py` on the *loader* output; per-channel noise σ sane; product mean
   reproduces ξᵢⱼ (extend the §14 check to the loader); **AND benchmark it/s with augment ON vs OFF**
   (≥3 runs, load recorded) — if ON ≪ OFF, fix the placement before training (§7).
3. **GATE B — cosmology-dependence (NEW, decisive):** confirm the cross statistics **vary with
   cosmology** across the TFDS (the full multi-cosmo set now allows this — the fiducial-only check
   could not). If they don't move with θ, the channels carry no info — stop and debug.
4. **Compress + train** the matrix (3 seeds, jaxili MAF; campaign-exact CNN). Cache compressed
   datasets; reuse across seeds.
5. **GATE C — calibration:** TARP-DRP (varied-θ, stratified by FoM3 tercile), SBC, L-C2ST — BEFORE
   trusting any contour (the leaky run *passed* calibration because the leak was self-consistent; here
   we expect calibration too, but verify). Use the existing Phase-D scripts.
6. **Contours + comparison:** per-patch geometry sweep (`geometry_resample.py` pattern), σ(w0)/2D/FoM3
   vs auto-only and vs the full-sphere result; the no-smoothing corners (`corner_resample.py` pattern).
7. **Write up** + update memory + felt.

---

## 7. Performance — keep the GPU fed, don't bottleneck on CPU FFTs

The cross augmentation is cheap *per map* (FFT of 80px × 6 pairs) but, done wrong, it **starves the
GPU**. The project has already paid for this: the harmonic-TFRecord loader ran ~1 it/s under CPU load
(`project_harmonic_tfrecord_training_path`), and `tfds.load` interleave defaults collapse 16→1 it/s
after the shuffle buffer drains (`project_tfds_load_interleave_tuning`). Design for throughput from
the start (`cluster-resources`, `coding-guidelines`):

- **Compute the cross on-DEVICE, batched — not per-example in a CPU `tf.data` map.** Let `tf.data`
  deliver only the **4 auto channels** (light), then compute the 6 cross channels batched on the GPU
  in the consumer: **JAX** `jnp.fft` inside the CNN compressor input step, **torch** `torch.fft`
  right before the wavelet-ℓ₁ (the L1 summary already runs on GPU). This keeps the FFTs on the A100
  and the CPU free for I/O. The old `_compute_cross_maps_tf` (CPU `tf.data`) is the anti-pattern.
- **Tune the loader:** `tfds.ReadConfig(interleave_cycle_length=8, interleave_block_length=16)` +
  `prefetch(AUTOTUNE)` + a real shuffle buffer; `num_parallel_calls=AUTOTUNE` for any remaining map.
- **One-time precompute is the fallback, not the default.** If on-device augmentation still
  bottlenecks, precompute the 10-ch flat-sky cross to a cache once (batched on GPU) and train off it
  — but that's ~245 GB and a build step, so only if the benchmark forces it.
- **BENCHMARK before scaling (GATE):** measure it/s with cross-augmentation ON vs OFF, on-node, with
  load + co-tenant recorded, ≥3 runs (`feedback_benchmark_dont_assume`,
  `feedback_dont_guess_time_estimates`). If augment-ON it/s ≪ augment-OFF, the GPU is starving — fix
  the placement before training. Do NOT state throughput you haven't measured.
- **GPU:** **GPU 1 only** (`feedback_gpu1_only`); sole tenant ⇒ `XLA_PYTHON_CLIENT_MEM_FRACTION` up
  to ~1.0; pick a batch size that saturates the A100. **CPU:** the loader threads are GIL-bound —
  more `--harmonic-loader-threads` did NOT help before (`project_tfdata_cross_route_leakage`); rely
  on `tf.data`'s C++ parallelism + on-device FFTs, not Python threads.
- Detached jobs: `setsid nohup … &` + poll with `pgrep`/log-grep (shell `wait` returns early under
  setsid; never `pkill -f` a self-matching pattern — `feedback_no_pkill_self_match`).
- Checkpoint-reload gotcha: jaxili truncates the high-dim Standardizer in hparams.json → L1 (2000-d)
  must RETRAIN, CNN (10-d) reloads bit-exact with an ABSOLUTE path
  (`reference_jaxili_checkpoint_reload_truncation`).

---

## 8. Guardrails (carry-forward lessons — violating these is how we get wrong conclusions)

- **Patch-local only.** Cross built from the patch's own autos. NEVER the full-sphere route (leakage).
- **Per-channel noise** for SNR (not shared auto-σ). Verify in stdout/meta.
- **Never PCA L1** (`--pca-components 0`); verify `pca_applied: False`.
- **Don't headline FoM3**; lead with σ(w0) + 2D areas.
- **Calibrate before contours.** Tightness ≠ correctness.
- **Example-disjoint compressor/NDE split by perm.**
- **One apodized-circular convolution definition**; product is pointwise. Beware lag/registration.
- **Patches are not independent for cross** (shared modes); this is a *methods comparison*, not a
  survey forecast — don't over-claim absolute constraining power.
- **Same auto channels across all arms** (the only thing that changes is the cross set).
- Report in the SKILL.md structure (Objective / Config fingerprint / Quant outcomes / Robustness /
  Conclusion / Next action). Apples-to-apples, reproducible, seed-stable.

---

## 9. Open design questions to settle early in the new session

1. Per-channel **per-scale** noise (option 2.1) vs per-channel-only (2.2) for the first pass?
2. Apodize the **product** map (×W²) or not?
3. Use a 4-ch auto TFDS vs slicing 0–3 of the 10-ch TFDS (whichever is cleaner/faster)?
4. Scale set: reuse auto `n_scales` as-is, or add a coarser scale for the smooth convolution cross?
5. Backlog (only if time): the **scale-matched product** `[ψₛκᵢ][ψₛκⱼ]` per (pair,scale) (notes §10).

## 10. Pointers

- Design/validation: `FLATSKY_CROSS_REDESIGN_NOTES.md`; leakage: `CROSS_MAP_LEAKAGE_FINDING.md`.
- Prior campaign result + diagnostics: `…/definitive_comparison_10deg/phase_c/analysis/SUMMARY_PHASE_D.md`.
- Reusable code: `_compute_cross_maps_{np,tf}` (L1 scripts), `validate_flatsky_cross.py`,
  `geometry_resample.py`, `corner_resample.py`, Phase-D calibration scripts (TARP/SBC/L-C2ST).
- Data: TFDS `grid_10deg_80px_nonoverlap180`; fiducial cache `full_sphere_cache_fiducial_10deg`.
- Memory index: `…/memory/MEMORY.md` (leakage, fom3-fragile, never-pca-l1, gpu1, noise-model, etc.).
