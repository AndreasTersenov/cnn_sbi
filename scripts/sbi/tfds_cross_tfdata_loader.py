"""Standard `tfds.load` + tf.data loader for the TFRecord cross dataset.

Mirrors the auto-only `_dataset_iter` in npe_cnn_nbody_tomo.py (the proven ~27 it/s path):
`tfds.load -> repeat -> shuffle -> map -> batch -> prefetch -> tfds.as_numpy`. The only
differences from auto-only are the dataset (10-channel cross, TFRecord) and the transform
(channel slice/scale, flip, theta H0->h0) — no per-batch noise (baked into the cache), no
hand-rolled interleave, no DLPack, no Grain worker processes.

Yields `{"maps":[B,H,W,C], "theta":[B,6]}` numpy batches — same contract as the other
loaders. Requires the cross dataset built with --file_format=tfrecord (ArrayRecord does
NOT support as_dataset / the tf.data path).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

N_CHANNELS = 10


def build_tfds_tfdata_iterator(
    tfds_name: str,
    data_dir: str,
    split: str,
    batch_size: int,
    seed: int,
    flip: bool,
    channel_scale: Optional[np.ndarray] = None,
    channel_slice: Optional[slice] = None,
    shuffle_buffer: int = 4096,
) -> Iterator[Dict[str, np.ndarray]]:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tf_dataset_nbody_tomo_cross as _cross_builder  # noqa: F401  (registers the dataset)

    split = {"val": "test"}.get(split, split)
    lo = (channel_slice.start or 0) if channel_slice is not None else 0
    hi = (channel_slice.stop or N_CHANNELS) if channel_slice is not None else N_CHANNELS

    # channel_scale is ALREADY sliced to the active channels by npe_cnn
    # (auto_cross -> [10], cross_only -> [6], auto_only -> [4]). Apply directly;
    # do NOT re-slice with [lo:hi] (that was a latent double-slice bug for
    # non-default channel_mode -> ValueError "shapes [H,W,6] [2]").
    scale_t = None
    if channel_scale is not None:
        scale_t = tf.constant(np.asarray(channel_scale, np.float32), dtype=tf.float32)
    # theta[3] = H0/100 -> h0, done as a graph-friendly elementwise multiply.
    h0_vec = tf.constant([1.0, 1.0, 1.0, 0.01, 1.0, 1.0], dtype=tf.float32)

    def _transform(ex):
        m = ex["map_nbody"][:, :, lo:hi]
        if scale_t is not None:
            # channel_scale is the per-channel RMS; DIVIDE to normalize each
            # channel to ~unit RMS. This mirrors the .npz reference path
            # (build_harmonic_batch_iterator: `maps / channel_scale`). Multiplying
            # here collapsed the ~1e-7 cross channels to ~1e-14 -- a silent
            # science bug caught by the 2026-05-30 batch-parity gate.
            m = m / scale_t
        if flip:
            m = tf.image.random_flip_left_right(m)
            m = tf.image.random_flip_up_down(m)
        theta = tf.cast(ex["theta"], tf.float32) * h0_vec
        return {"maps": m, "theta": theta}

    # Cap the per-shard interleave fan-out. With the full dataset (~2112 shards),
    # the tfds default lets tf.data interleave across ALL shards, which starves
    # the shuffle buffer after the initial fill (measured: ~15 it/s for 50 steps
    # then collapses to ~1 it/s with GPU 0%). cycle_length=8 + block_length=16
    # bounds the read fan-out so the shuffle buffer refills steadily.
    read_config = tfds.ReadConfig(
        interleave_cycle_length=8,
        interleave_block_length=16,
        shuffle_seed=seed,
    )
    ds = tfds.load(tfds_name, split=split, data_dir=data_dir, read_config=read_config)
    ds = ds.repeat().shuffle(shuffle_buffer, seed=seed)
    ds = ds.map(_transform, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return iter(tfds.as_numpy(ds))
