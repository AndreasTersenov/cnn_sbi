---
name: Compressor returns last-step params, not best-val — every kept iter is 0.27-0.34 nats below its own best
status: open
tags:
    - cnn-auto-push
created-at: 2026-05-18T21:32:26.350207769Z
outcome: Filed pending Stage-B-with-best-val test. Cheap rerun on iter-5 (params_nd_compressor_batch48000.pkl already saved) would falsify or confirm; F1 in audits/2026-05-18_A2_loss_curves/README.md
---

Surfaced 2026-05-18 ~21:30 UTC by PHASE AUDIT A2 (Ralph iter-2 while iter-16 trains). The compressor's `train_compressor_vmim` (`scripts/sbi/npe_cnn_nbody_tomo.py:1736-2270`) ends with `return params_merged, state_cnn` — i.e. the **final-step** parameters — and has NO `best_step` / `best_val_loss` tracking. Compare with the NDE flow (`:2748-2841`) which DOES early-stop on best-val. The downstream cache-fingerprint helper (`:3475-3486`) picks `final_params[-1]` (last-step) to keep the cache deterministic, reinforcing the policy.

The screening-budget data shows this matters across the kept iterations:

| iter   | best val (step)    | final val (step 60k) | nats wasted |
|--------|--------------------|----------------------|-------------|
| iter-5 | -12.72 (step 48k)  | -12.44               | **0.28**    |
| iter-14| -12.45 (step 33k)  | -12.18               | **0.27**    |
| iter-15| -11.49 (step 51k)  | -11.15               | **0.34**    |

Three reasons this matters NOW:

1. **iter-14 (wider conv) is overtraining hard** by step 33k (55%). Its TIE status (`+1.4% within 0.5σ noise floor`) is the LAST-step evaluation; the best-val FoM3 might have been a genuine keep. The "wider conv hurts at this LR" conclusion ([[results.tsv]] row 14) may not be the real story.
2. **iter-15 (resnet50_gn) is being read at step 60k, past its argmin at 51k.** The collapse magnitude vs iter-5 (-52.8%) was computed against a 60k-final compressor that was already past its peak. The undertraining story remains true (resnet50_gn at step 60k is much shallower than plain anywhere on its curve), but the magnitude is partially F1-amplified.
3. **iter-16 (Q2 in flight, 120k)** will be read at step 120k. If 120k argmin is at, say, step 96k (80% by analogy), iter-16 final could be 0.3+ nats looser than iter-16 best. Q2 effect could be **underestimated**. Worse: if 120k overtrains relative to 60k, iter-16 final could even regress below iter-5 final while iter-16 best is much better. **The audit test must include "iter-16 with best-val ckpt" alongside "iter-16 with last-step ckpt"**, else we cannot orthogonalize Q2 from F1.

## Proposed cheap test (Q2b)

The best-val checkpoint for iter-5 (`params_nd_compressor_batch48000.pkl`) is already on disk at:
```
/nas/tersenov/claude-notes/runs/cnn-auto-push-18-20-2026/iter-5/
  compressor/nobnt/vmim/nbody/sigma_0.26/gal_density_30/bin_4/
  params_nd_compressor_batch48000.pkl
  opt_state_resnet_batch48000.pkl
```

Rerun Stage B (NDE training + posterior sampling for 3 seeds) using this checkpoint. ~3-10 min wall per seed; can be parallelized. Compare FoM3 to iter-5 (final-step ckpt) = 18 568. If FoM3 jumps to ~20-22k from best-val alone, F1 is confirmed as a dominant lever AND a near-free improvement is in hand.

Recipe (per seed, single GPU):
```bash
conda run -n jaxili python scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices <gpu> --seed <SEED> \
  --cache-dir <iter-5b>/cache/seed_<SEED> \
  --save-dir <iter-5b>/eval/seed_<SEED> \
  --compressor-params <iter-5>/compressor/.../params_nd_compressor_batch48000.pkl \
  --compressor-state  <iter-5>/compressor/.../opt_state_resnet_batch48000.pkl \
  --total-steps 10000 --save-every 500 --patience 50 --batch-size 256 \
  --nvp-layers 8 --nvp-hidden 256 \
  --summary-clip-value 5.0 --npe-samples 100000 \
  --posterior-out <iter-5b>/posteriors/cnn_auto_plain_step60000_s<SEED>.npy \
  --ds-batch-size 500 --no-standardize-summary \
  --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --zero-mean-maps --compressor-arch plain \
  --compressor-dim 16 --compressor-dense-width 512 \
  --compressor-conv-channels 64,128,256 \
  --compressor-pool-window 16 --compressor-pool-stride 8
```

Each seed needs its own `--cache-dir` (different compressed-dataset fingerprint). `--no-train-compressor` is implicit when `--compressor-params` is given.

## Larger fix (not in scope for this Ralph iteration)

Patch `train_compressor_vmim` to track `best_val_loss` / `best_step` and return the best-val params. This would require: (a) holding two copies of params in RAM during training (~25 MB for plain, ~100 MB for resnet50_gn — negligible); (b) updating the cache-fingerprint helper at `:3475-3486` to use the saved best-val step instead of `final_params[-1]`. Out of scope for an audit iteration; file an Andreas-authorization question if the cheap test confirms F1 is dominant.

## Related

- [[cnn-auto-resnet50gn-undertrained]] — F1 partially explains the iter-15 collapse magnitude (not the qualitative story).
- [[cnn-auto-compressor-undertrained]] — pre-existing finding that 60k is not plateaued. Combined with F1: 60k both undertrains AND uses an overtrained checkpoint relative to its own best. Two distinct levers.
- [[cnn-auto-lr-schedule-shape]] — the piecewise LR schedule decays to 4% by step 2/3·total_steps, which shapes the location of the argmin (always near 80% of training, where the polish phase begins).
