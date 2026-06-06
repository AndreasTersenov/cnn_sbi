"""Standard `tfds.load` + tf.data loader for the TFRecord cross dataset.

Mirrors the auto-only `_dataset_iter` in npe_cnn_nbody_tomo.py (the proven ~27 it/s path):
`tfds.load -> repeat -> shuffle -> map -> batch -> prefetch -> tfds.as_numpy`. The only
differences from auto-only are the dataset (10-channel cross, TFRecord) and the transform
(channel slice/scale, flip, theta H0->h0) — no per-batch noise (baked into the cache), no
hand-rolled interleave, no DLPack, no Grain worker processes.

Yields `{"maps":[B,H,W,C], "theta":[B,6]}` numpy batches — same contract as the other
loaders. Requires the cross dataset built with --file_format=tfrecord (ArrayRecord does
NOT support as_dataset / the tf.data path).

`perm_lo`/`perm_hi` (added 2026-06-05 for the 10deg campaign) filter examples by `perm`
BEFORE shuffle, giving an EXAMPLE-disjoint compressor<->NDE split (compressor perms 0-4,
NDE perms 5-6) that keeps all cosmologies in both — see PLAN_PHASE_B.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

N_CHANNELS = 10


def _build_cross_ds(
    tfds_name: str,
    data_dir: str,
    split: str,
    seed: int,
    flip: bool,
    channel_scale: Optional[np.ndarray],
    channel_slice: Optional[slice],
    perm_lo: Optional[int],
    perm_hi: Optional[int],
    *,
    repeat: bool,
    shuffle: bool,
    batch_size: int,
    drop_remainder: bool,
    shuffle_buffer: int = 4096,
):
    """Shared (tf.data) pipeline for the cross TFDS.

    `repeat`/`shuffle` True -> infinite training stream; both False -> a single
    deterministic finite pass (for compression and channel-RMS estimation).
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tf_dataset_nbody_tomo_cross as _cross_builder  # noqa: F401 (registers the dataset)

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

    # Cap the per-shard interleave fan-out. With the full dataset (~2048 shards),
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
    if perm_lo is not None or perm_hi is not None:
        plo = 0 if perm_lo is None else int(perm_lo)
        phi = 1_000_000 if perm_hi is None else int(perm_hi)
        ds = ds.filter(
            lambda ex: tf.logical_and(ex["perm"] >= plo, ex["perm"] <= phi)
        )
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.map(_transform, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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
    perm_lo: Optional[int] = None,
    perm_hi: Optional[int] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    """Infinite shuffled `{"maps","theta"}` batches for compressor training."""
    import tensorflow_datasets as tfds

    ds = _build_cross_ds(
        tfds_name, data_dir, split, seed, flip, channel_scale, channel_slice,
        perm_lo, perm_hi,
        repeat=True, shuffle=True, batch_size=batch_size, drop_remainder=True,
        shuffle_buffer=shuffle_buffer,
    )
    return iter(tfds.as_numpy(ds))


def iter_cross_tfds_batches(
    tfds_name: str,
    data_dir: str,
    split: str,
    batch_size: int,
    seed: int,
    flip: bool,
    channel_scale: Optional[np.ndarray] = None,
    channel_slice: Optional[slice] = None,
    perm_lo: Optional[int] = None,
    perm_hi: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """One finite deterministic pass: yields (maps[B,H,W,C], theta[B,6]) numpy.

    maps are already channel-sliced + scale-divided + (optionally flipped); theta is
    already H0->h0 converted. Used by compress_dataset_from_cross_tfds.
    """
    import tensorflow_datasets as tfds

    ds = _build_cross_ds(
        tfds_name, data_dir, split, seed, flip, channel_scale, channel_slice,
        perm_lo, perm_hi,
        repeat=False, shuffle=False, batch_size=batch_size, drop_remainder=False,
    )
    for b in tfds.as_numpy(ds):
        yield b["maps"], b["theta"]


def compute_cross_tfds_channel_rms(
    tfds_name: str,
    data_dir: str,
    split: str = "train",
    n_sample: int = 8000,
    channel_slice: Optional[slice] = None,
    perm_lo: Optional[int] = None,
    perm_hi: Optional[int] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-channel RMS (sqrt(mean(x^2))) over the ACTIVE (sliced) channels, from a
    finite `n_sample`-example sample of `split`. Maps are NOT scaled here (channel_scale
    is None) -- this returns the scale itself. Matches the harmonic cache RMS semantics:
    the returned vector has length = active channel count and is the divisor downstream.
    """
    import tensorflow_datasets as tfds

    ds = _build_cross_ds(
        tfds_name, data_dir, split, seed=0, flip=False,
        channel_scale=None, channel_slice=channel_slice,
        perm_lo=perm_lo, perm_hi=perm_hi,
        repeat=False, shuffle=False, batch_size=batch_size, drop_remainder=False,
    )
    sum_sq = None
    n_pixels = 0
    n_examples = 0
    for b in tfds.as_numpy(ds):
        m = b["maps"].astype(np.float64)  # (B,H,W,C)
        c = m.shape[-1]
        flat = m.reshape(-1, c)
        if sum_sq is None:
            sum_sq = np.zeros(c, dtype=np.float64)
        sum_sq += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]
        n_examples += m.shape[0]
        if n_examples >= n_sample:
            break
    if sum_sq is None or n_pixels == 0:
        raise RuntimeError(f"compute_cross_tfds_channel_rms: no examples for split={split}.")
    return np.sqrt(sum_sq / n_pixels).astype(np.float32)
