from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf


# BNT matrix used in existing tomographic BNT experiments in this repository.
BNT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [0.4521097, -1.4521097, 1.0, 0.0],
        [0.0, 0.25127807, -1.251278, 1.0],
    ],
    dtype=np.float32,
)

BNT_MATRIX_VERSION = "tomo4_bnt_v1"


def validate_bnt_configuration(nbins: int, tomo_bin_indices: Sequence[int]) -> None:
    bins = tuple(int(b) for b in tomo_bin_indices)
    if nbins != 4 or bins != (1, 2, 3, 4):
        raise ValueError(
            "--apply-bnt requires full tomography: --nbins 4 and "
            "--tomo-bin-indices 1,2,3,4."
        )


def apply_bnt_numpy(maps: np.ndarray) -> np.ndarray:
    arr = np.asarray(maps, dtype=np.float32)
    if arr.shape[-1] != 4:
        raise ValueError(f"BNT expects 4 channels, got shape {arr.shape}.")
    transformed = np.tensordot(arr, BNT_MATRIX, axes=([-1], [1]))
    return transformed.astype(np.float32)


def apply_bnt_tf(maps: tf.Tensor) -> tf.Tensor:
    if maps.shape.rank is not None and maps.shape[-1] is not None:
        if int(maps.shape[-1]) != 4:
            raise ValueError(
                f"BNT expects 4 channels, got last axis={int(maps.shape[-1])}."
            )
    matrix = tf.constant(BNT_MATRIX, dtype=maps.dtype)
    return tf.tensordot(maps, matrix, axes=[[-1], [1]])
