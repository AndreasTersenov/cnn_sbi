#!/usr/bin/env python
"""Profile the harmonic TFRecord -> training data path, stage by stage.

The smoke showed ~7 it/s with the GPU ~90% idle, so the ceiling is host-side,
not GPU compute. This decomposes the per-batch host cost into:

  1. pipeline base   : files -> interleave -> parse(decode_raw+reshape) -> batch
  2. + shuffle       : shard shuffle + 4096-buffer shuffle
  3. + in-graph flip : tf.image random LR/UD per patch
  4. + .numpy()      : tf CPU tensor -> numpy (host copy)  [== reader output]
  5. + H2D           : jnp.asarray + block_until_ready (host -> GPU 1)
  6. batched-parse   : alternative -- batch raw records, vectorized parse_example

Each stage's marginal ms/batch is reported. Stage 4 rate is the decisive
"reader-only" number: if it is close to the observed ~7 it/s, the tf.data
pipeline itself is the wall (optimize the pipeline / parsing); if it is much
higher, the wall is the un-overlapped consumer side (use device prefetch).

Run on GPU 1 (for the H2D stage):
    conda run -n jaxili python scripts/sbi/tests/profile_tfrecord_pipeline.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # GPU 1 only (project rule)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

SBI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SBI_DIR))
import npe_cnn_nbody_tomo as npe  # noqa: E402

TFR = Path("/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid")
REGIME = "nobnt"
BATCH = 128
NREAL = 50
NWARM = 12
NTIME = 200
COMP = ""  # NONE

FEAT = {
    "patch": tf.io.FixedLenFeature([], tf.string),
    "theta": tf.io.FixedLenFeature([], tf.string),
}

shards = npe._list_harmonic_tfrecord_shards(TFR, REGIME, "train")[:NREAL]
SHARD_PATHS = [str(p) for p in shards]


def build_ds(shuffle: bool, flip: bool):
    def parse(raw):
        ex = tf.io.parse_single_example(raw, FEAT)
        patch = tf.reshape(tf.io.decode_raw(ex["patch"], tf.float32), (160, 160, 10))
        if flip:
            patch = tf.image.random_flip_left_right(patch)
            patch = tf.image.random_flip_up_down(patch)
        theta = tf.reshape(tf.io.decode_raw(ex["theta"], tf.float64), (6,))
        return patch, theta

    ds = tf.data.Dataset.from_tensor_slices(SHARD_PATHS)
    if shuffle:
        ds = ds.shuffle(len(SHARD_PATHS), seed=0)
    ds = ds.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=COMP),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    if shuffle:
        ds = ds.shuffle(4096, seed=0)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)
    return ds


def build_ds_batched_parse(shuffle: bool, flip: bool):
    """Alternative: batch raw serialized records, then vectorized parse_example."""
    feat_batched = {
        "patch": tf.io.FixedLenFeature([], tf.string),
        "theta": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_batch(raw_batch):
        ex = tf.io.parse_example(raw_batch, feat_batched)  # vectorized over batch
        patch = tf.io.decode_raw(ex["patch"], tf.float32)  # (B, 160*160*10)
        patch = tf.reshape(patch, (-1, 160, 160, 10))
        if flip:
            patch = tf.image.random_flip_left_right(patch)
            patch = tf.image.random_flip_up_down(patch)
        theta = tf.reshape(tf.io.decode_raw(ex["theta"], tf.float64), (-1, 6))
        return patch, theta

    ds = tf.data.Dataset.from_tensor_slices(SHARD_PATHS)
    if shuffle:
        ds = ds.shuffle(len(SHARD_PATHS), seed=0)
    ds = ds.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=COMP),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    if shuffle:
        ds = ds.shuffle(4096, seed=0)
    ds = ds.batch(BATCH, drop_remainder=True)
    ds = ds.map(parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat().prefetch(tf.data.AUTOTUNE)
    return ds


def time_rate(make_iter, consume):
    it = make_iter()
    for _ in range(NWARM):
        consume(next(it))
    t0 = time.time()
    for _ in range(NTIME):
        consume(next(it))
    dt = time.time() - t0
    return NTIME / dt, dt


def main():
    import jax
    import jax.numpy as jnp

    print(f"  jax devices: {jax.devices()}")
    print(f"  {NREAL} realizations, batch {BATCH}, {NTIME} timed batches\n")

    results = {}

    # Stage 1: pipeline base (no shuffle, no flip), consume tf tensor (no host copy)
    results["1_base"] = time_rate(
        lambda: iter(build_ds(shuffle=False, flip=False)),
        lambda b: b[0],  # touch tensor; no .numpy()
    )
    # Stage 2: + shuffle
    results["2_+shuffle"] = time_rate(
        lambda: iter(build_ds(shuffle=True, flip=False)),
        lambda b: b[0],
    )
    # Stage 3: + flip
    results["3_+flip"] = time_rate(
        lambda: iter(build_ds(shuffle=True, flip=True)),
        lambda b: b[0],
    )
    # Stage 4: + .numpy()  (== reader output to host)
    results["4_+numpy(reader)"] = time_rate(
        lambda: iter(build_ds(shuffle=True, flip=True)),
        lambda b: b[0].numpy(),
    )
    # Stage 5: + H2D (jnp.asarray + block)
    def consume_h2d(b):
        x = jnp.asarray(b[0].numpy())
        jax.block_until_ready(x)

    results["5_+H2D"] = time_rate(
        lambda: iter(build_ds(shuffle=True, flip=True)),
        consume_h2d,
    )
    # Stage 6: batched vectorized parse (+numpy, for comparison vs stage 4)
    results["6_batchedparse+numpy"] = time_rate(
        lambda: iter(build_ds_batched_parse(shuffle=True, flip=True)),
        lambda b: b[0].numpy(),
    )

    print(f"  {'stage':<26}{'it/s':>8}{'ms/batch':>11}{'marginal ms':>13}")
    order = ["1_base", "2_+shuffle", "3_+flip", "4_+numpy(reader)", "5_+H2D"]
    prev_ms = 0.0
    for k in order:
        rate, _ = results[k]
        ms = 1000.0 / rate
        marg = ms - prev_ms
        print(f"  {k:<26}{rate:>8.1f}{ms:>11.1f}{marg:>13.1f}")
        prev_ms = ms
    rb, _ = results["6_batchedparse+numpy"]
    print(f"\n  ALT: {('6_batchedparse+numpy'):<22}{rb:>8.1f}{1000.0/rb:>11.1f}"
          f"   (vs stage 4 = {results['4_+numpy(reader)'][0]:.1f} it/s)")
    print(
        "\n  Decisive: stage-4 'reader-only' rate vs observed ~7 it/s training.\n"
        "  If stage-4 ~= 7-9 -> pipeline-bound (parsing/copies). If >> 7 ->\n"
        "  consumer-bound (un-overlapped H2D/step) -> device prefetch wins."
    )


if __name__ == "__main__":
    main()
