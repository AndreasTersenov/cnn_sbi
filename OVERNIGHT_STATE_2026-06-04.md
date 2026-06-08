# Overnight state — 10° dataset build (2026-06-04 ~04:46 UTC)

## STATUS UPDATE (~12:xx UTC): SHT cache DONE; TFDS reserialize running (slow, node contention)
- **Done + verified:** archival→/nas (deleted local, 0-diff verified); 10° **fiducial cache** (200
  perms, kept: `cross_maps_campaign/full_sphere_cache_fiducial_10deg`); 10° **grid cache** (9093 npz,
  transient, intact at `…/full_sphere_cache_grid_10deg`). Smoke + GRID verify both passed.
- **Running:** `run_10deg_tfds_resume.sh` → reserialize grid cache → TFDS (TFRecord) at
  `/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`,
  then verify (count + bit-exact) → delete transient cache → HANDOFF "READY". Watcher `b2fqvzxsw`.
- **Why slow (~9 h):** another user (`titan`) saturates the node (~80/128 cores, load ~155); the
  reserialize runs ~50 examples/s on the ~48 spare cores (1.64M examples total). Faster if titan frees up.
- **3 build bugs fixed tonight** (all in my code, no installs): (a) `tfds` CLI needs `apache_beam`
  → build programmatically via `build_10deg_tfds.py`; (b) empty `obs` split → builder skips empty
  splits; (c) missing `if __name__=="__main__"` guard → spawn Pool recursively re-ran the build.
  Build is correct + progressing now. Lean config: 8 CPU-only (`CUDA_VISIBLE_DEVICES=`),
  1-thread workers (polite to the shared node).
- Resume can be re-run safely if interrupted (idempotent; grid cache intact). Logs:
  `cross_maps_campaign/run_10deg_tfds_resume.log`.

---


Autonomous overnight run launched while Andreas sleeps. Goal: produce + verify the 10° TFDS
dataset (PLAN_10DEG_CAMPAIGN.md). **Stop before** the L1 loader + campaign training (morning, with
Andreas) — L1 datavector is science-critical (channel-noise / no-PCA rules; past 4× FoM bugs).

## What is running
1. **Archival rsyncs** (→ `/nas/tersenov/archive_20deg/`): 20° harmonic cache (623 GB) + 20° TFDS
   (auto 275 GB + cross 421 GB). Frees disk for the 10° build. ~5 h. Logs:
   `…/cross_maps_campaign/rsync_archive_{20deg,tfds}.log`.
2. **`run_10deg_phase0.sh`** (setsid-detached): waits for the archival → **VERIFIES** (rsync itemize
   = 0 files differing; fail-safe) → deletes local 20° (recoverable on /nas) → launches the build.
   Log: `…/cross_maps_campaign/run_10deg_phase0.log`.
3. **`run_10deg_build.sh`** (chained by phase0): smoke (fiducial 3-perm) → fiducial cache (200 perm,
   KEPT) → grid cache (transient) → 10° cross TFDS → verify (count + bit-exact) → delete transient
   cache → writes **`HANDOFF_10DEG.md`**. HARD GATES abort-and-log; never deletes/keeps garbage.
4. **Watcher `byrqv1j76`** (harness bg): notifies me when phase0 exits (done OR abort).

## Code written this session (all py_compile / bash -n clean)
- `tf_dataset_nbody_tomo.py`: `_build_non_overlapping_centers(..., max_abs_lat)` — polar-safe centers
  (tested: 180 centers, |lat|<65°, min-sep 14.2°; 20° configs unchanged).
- `build_full_sphere_cross_cache.py`: `--max-abs-lat` wired through BuildConfig + center call.
- `tf_dataset_nbody_tomo_cross.py`: config `grid_10deg_80px_nonoverlap180` + `CROSS_TFDS_CACHE_DIR` env.
- `run_10deg_{phase0,build}.sh` (the orchestrators).

## 10° geometry (locked)
field 10°, 80px (7.5 arcmin/px = same res as 20°), 180 non-overlap patches |lat|<75°, nside=512,
lmax=1024, σe=0.26, density=10, nobnt. Grid = `--cosmo-subset grid --realizations 0,1,2,3,4,5,6`.

## Morning work (NOT done autonomously)
1. **L1-reads-TFDS loader** (science-critical: `channel_empirical_global` noise, **PCA OFF / never
   PCA L1**; verify datavector vs a reference). 2. CNN loader `read_config` retune for 10° shards.
3. Clean compressor/NDE split by **cosmo_idx** (e.g. 1–630 vs 631–899) — kills the 20° leakage.
4. Run the 4 arms + the full diagnostic suite (geometry map, spread, bias/error-budget, SBC, CNN
   L-C2ST). **Headline test: does L1's −0.37σ fiducial w0 offset shrink at 10°?** (flat-sky).

## If something aborted
Read `run_10deg_phase0.log` + `run_10deg_build.log` + `HANDOFF_10DEG.md`. Deletes are gated on
verify (0 rsync diffs), so a failed verify leaves the 20° local intact. 20° is on /nas regardless.
