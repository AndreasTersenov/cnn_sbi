#!/usr/bin/env python
"""Build the 10-channel auto+cross TFDS dataset (reserializes the .npz cache).

Programmatic wrapper around `NbodyCosmogridDatasetTomoCross.download_and_prepare` so we
control file_format + an optional subset cap without relying on the tfds CLI.

  # benchmark-sized subset (ArrayRecord):
  python build_cross_tfds_dataset.py --data-dir /nas/tersenov/tfds_cross_arrayrecord \
    --file-format array_record --cosmo-limit 50
  # full build:
  python build_cross_tfds_dataset.py --data-dir /nas/tersenov/tfds_cross_arrayrecord \
    --file-format array_record

CPU/IO only (no GPU). The build is bottlenecked decompressing the zlib .npz cache on one
core (~17-20 examples/s serial); use --cosmo-limit for a quick subset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--file-format", default="array_record",
                    choices=["array_record", "tfrecord"])
    ap.add_argument("--cosmo-limit", type=int, default=0,
                    help="distinct cosmologies/split (0 = full grid).")
    ap.add_argument("--config", default="grid_20deg_160px_nonoverlap48")
    args = ap.parse_args()

    if args.cosmo_limit > 0:
        os.environ["CROSS_TFDS_COSMO_LIMIT"] = str(args.cosmo_limit)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import tf_dataset_nbody_tomo_cross as M

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    print(f"building {args.config} -> {args.data_dir} [{args.file_format}] "
          f"cosmo_limit={args.cosmo_limit or 'full'}", flush=True)
    t0 = time.time()
    builder = M.NbodyCosmogridDatasetTomoCross(
        config=args.config, data_dir=args.data_dir, file_format=args.file_format,
    )
    builder.download_and_prepare()
    print(f"DONE in {time.time() - t0:.0f}s -> {args.data_dir} [{args.file_format}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
