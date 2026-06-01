"""Pure cross-loader throughput probe (NO training loop, NO GPU).

Times build_tfds_tfdata_iterator's batch production from the /nas FUSE dataset, to
localize the bottleneck: if pure loading sustains ~16 it/s, the cap is the training
loop; if it collapses to ~1 it/s, the data path (FUSE/mergerfs) is the bottleneck.

CPU-only; records loadavg so the number is interpretable under contention.
"""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path
import numpy as np
_SBI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SBI))

TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/nas/tersenov/tfds_cross_tfrecord_full")
    ap.add_argument("--n-batches", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()
    from tfds_cross_tfdata_loader import build_tfds_tfdata_iterator
    # cross_only slice (6 ch) + the cached cross RMS, matching the training feed.
    scale = np.array([2.4576e-07, 3.4121e-07, 4.3173e-07, 5.8681e-07, 8.0807e-07, 1.6744e-06], np.float32)
    it = build_tfds_tfdata_iterator(
        tfds_name=TFDS_NAME, data_dir=args.data_dir, split="train",
        batch_size=args.batch_size, seed=41, flip=True,
        channel_scale=scale, channel_slice=slice(4, 10), shuffle_buffer=4096,
    )
    print(f"data_dir={args.data_dir}  batch={args.batch_size}  loadavg={os.getloadavg()[0]:.1f}")
    t0 = time.time(); t_win = t0; bytes_win = 0
    for i in range(1, args.n_batches + 1):
        b = next(it)
        bytes_win += b["maps"].nbytes
        if i % args.report_every == 0:
            now = time.time()
            win_rate = args.report_every / (now - t_win)
            mb_s = bytes_win / (now - t_win) / 1e6
            print(f"  batch {i:>5}: window {win_rate:6.2f} it/s ({mb_s:6.1f} MB/s)  "
                  f"cum {i/(now-t0):6.2f} it/s  loadavg={os.getloadavg()[0]:.1f}", flush=True)
            t_win = now; bytes_win = 0
    total = time.time() - t0
    print(f"\nTOTAL: {args.n_batches} batches in {total:.1f}s = {args.n_batches/total:.2f} it/s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
