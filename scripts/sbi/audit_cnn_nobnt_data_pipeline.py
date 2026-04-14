#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tf_dataset_nbody_tomo as _tomo_builder  # noqa: F401
from npe_cnn_jaxili_nbody_tomo import build_augmentation, parse_tomo_bin_indices


def _split_size(tfds_name: str, split: str) -> int:
    ds = tfds.load(tfds_name, split=split)
    card = tf.data.experimental.cardinality(ds).numpy()
    if int(card) >= 0:
        return int(card)
    return int(sum(1 for _ in ds))


def _collect_ids_and_theta(
    tfds_name: str,
    split: str,
    theta_decimals: int = 8,
) -> Tuple[set[str], set[Tuple[float, ...]]]:
    ds = tfds.load(
        tfds_name,
        split=split,
        read_config=tfds.ReadConfig(add_tfds_id=True),
    )
    id_set: set[str] = set()
    theta_set: set[Tuple[float, ...]] = set()
    for idx, ex in enumerate(tfds.as_numpy(ds)):
        tfds_id = ex["tfds_id"]
        if isinstance(tfds_id, bytes):
            tfds_id = tfds_id.decode("utf-8")
        else:
            tfds_id = str(tfds_id)
        theta = tuple(np.round(ex["theta"].astype(np.float64), theta_decimals).tolist())
        id_set.add(tfds_id)
        theta_set.add(theta)
        if (idx + 1) % 50000 == 0:
            print(f"  [{split}] scanned {idx + 1} examples ...")
    return id_set, theta_set


def _run_augmentation_checks(
    tfds_name: str,
    split: str,
    map_kind: str,
    sigma_e: float,
    galaxy_density: float,
    field_size: int,
    field_npix: int,
    nbins: int,
    tomo_bin_indices: tuple[int, ...],
) -> Dict[str, object]:
    augmentation = build_augmentation(
        map_kind=map_kind,
        sigma_e=sigma_e,
        galaxy_density=galaxy_density,
        field_size=field_size,
        field_npix=field_npix,
        nbins=nbins,
        tomo_bin_indices=tomo_bin_indices,
        apply_bnt=False,
    )

    ds = tfds.load(tfds_name, split=split)
    raw_example = next(iter(ds.take(1)))

    theta_raw = raw_example["theta"].numpy().astype(np.float64)
    out1 = augmentation(raw_example)
    out2 = augmentation(raw_example)

    theta_aug = out1["theta"].numpy().astype(np.float64)
    maps1 = out1["maps"].numpy().astype(np.float64)
    maps2 = out2["maps"].numpy().astype(np.float64)

    expected = theta_raw.copy()
    expected[3] = expected[3] / 100.0
    theta_max_abs_err = float(np.max(np.abs(theta_aug - expected)))
    map_delta_l2 = float(np.linalg.norm(maps1 - maps2))

    return {
        "raw_theta_h0": float(theta_raw[3]),
        "aug_theta_h0": float(theta_aug[3]),
        "expected_aug_theta_h0": float(expected[3]),
        "theta_max_abs_error_vs_expected": theta_max_abs_err,
        "theta_h0_rescaled_pass": bool(theta_max_abs_err < 1e-6),
        "maps_shape": list(maps1.shape),
        "maps_are_finite_pass": bool(np.isfinite(maps1).all() and np.isfinite(maps2).all()),
        "augmentation_stochastic_l2": map_delta_l2,
        "augmentation_stochastic_pass": bool(map_delta_l2 > 0.0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static no-BNT CNN data-pipeline audit for split/leakage/preprocess checks."
    )
    p.add_argument(
        "--tfds-name",
        type=str,
        default="NbodyCosmogridDatasetTomo/grid_20deg_160px",
    )
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="test")
    p.add_argument("--split-a", type=str, default="train[:70%]")
    p.add_argument("--split-b", type=str, default="train[70%:]")
    p.add_argument("--map-kind", type=str, default="nbody")
    p.add_argument("--field-size", type=int, default=20)
    p.add_argument("--field-npix", type=int, default=160)
    p.add_argument("--sigma-e", type=float, default=0.26)
    p.add_argument("--galaxy-density", type=float, default=30 / 4)
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--tomo-bin-indices", type=str, default="1,2,3,4")
    p.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write audit JSON report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tomo_bin_indices = parse_tomo_bin_indices(args.tomo_bin_indices)
    if args.nbins != len(tomo_bin_indices):
        raise ValueError(
            f"nbins={args.nbins} does not match tomo bins {tomo_bin_indices}."
        )

    print("######## DATA SPLIT CARDINALITIES ########")
    split_sizes = {
        "train_split": int(_split_size(args.tfds_name, args.train_split)),
        "val_split": int(_split_size(args.tfds_name, args.val_split)),
        "split_a": int(_split_size(args.tfds_name, args.split_a)),
        "split_b": int(_split_size(args.tfds_name, args.split_b)),
    }
    for key, value in split_sizes.items():
        print(f"  {key}: {value}")

    print("######## SPLIT OVERLAP AUDIT (tfds_id + theta) ########")
    train_ids, train_theta = _collect_ids_and_theta(args.tfds_name, args.train_split)
    val_ids, val_theta = _collect_ids_and_theta(args.tfds_name, args.val_split)
    split_a_ids, split_a_theta = _collect_ids_and_theta(args.tfds_name, args.split_a)
    split_b_ids, split_b_theta = _collect_ids_and_theta(args.tfds_name, args.split_b)

    overlap = {
        "train_vs_val": {
            "tfds_id_overlap": int(len(train_ids & val_ids)),
            "theta_overlap": int(len(train_theta & val_theta)),
            "train_unique_theta": int(len(train_theta)),
            "val_unique_theta": int(len(val_theta)),
        },
        "split_a_vs_split_b": {
            "tfds_id_overlap": int(len(split_a_ids & split_b_ids)),
            "theta_overlap": int(len(split_a_theta & split_b_theta)),
            "split_a_unique_theta": int(len(split_a_theta)),
            "split_b_unique_theta": int(len(split_b_theta)),
        },
    }
    print(
        "  train_vs_val: "
        f"id_overlap={overlap['train_vs_val']['tfds_id_overlap']} "
        f"theta_overlap={overlap['train_vs_val']['theta_overlap']}"
    )
    print(
        "  split_a_vs_split_b: "
        f"id_overlap={overlap['split_a_vs_split_b']['tfds_id_overlap']} "
        f"theta_overlap={overlap['split_a_vs_split_b']['theta_overlap']}"
    )

    print("######## AUGMENTATION / PARAMETER PREPROCESS CHECKS ########")
    preprocess_checks = _run_augmentation_checks(
        tfds_name=args.tfds_name,
        split=args.train_split,
        map_kind=args.map_kind,
        sigma_e=args.sigma_e,
        galaxy_density=args.galaxy_density,
        field_size=args.field_size,
        field_npix=args.field_npix,
        nbins=args.nbins,
        tomo_bin_indices=tomo_bin_indices,
    )
    print(
        "  h0_rescaled_pass="
        f"{preprocess_checks['theta_h0_rescaled_pass']} "
        "augmentation_stochastic_pass="
        f"{preprocess_checks['augmentation_stochastic_pass']}"
    )

    payload = {
        "tfds_name": args.tfds_name,
        "splits": {
            "train_split": args.train_split,
            "val_split": args.val_split,
            "split_a": args.split_a,
            "split_b": args.split_b,
        },
        "split_sizes": split_sizes,
        "overlap": overlap,
        "preprocess_checks": preprocess_checks,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved audit report → {args.output_json}")


if __name__ == "__main__":
    main()
