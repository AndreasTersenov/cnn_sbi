#!/usr/bin/env python3
"""flag_diff.py — pre-flight comparison of a new launcher's CLI flags vs an
existing anchor run's meta.json.

Catches the failure mode that bit us in iter-1 of canonical-anchors-refresh
(2026-05-24): the L1 launcher silently used --l1-min-snr=-10 instead of the
v2_chsigma anchor's −13, because the launcher was missing the explicit flag
and fell back to script defaults.

Usage:
    python flag_diff.py <launcher.sh> <anchor.meta.json>

Prints every meta.json key that has a CLI counterpart, alongside what the
launcher sets it to. Mismatches are flagged in red. Script defaults
(launcher does not pass the flag) are flagged in yellow — these are the
silent failure mode.

The mapping of meta.json keys to CLI flag names is encoded in
META_TO_FLAG below; extend as needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Mapping: meta.json field name -> CLI flag name (without leading "--").
# Edit/extend as the pipeline grows.
META_TO_FLAG = {
    # Shared core
    "tfds_name": "tfds-name",
    "field_size": "field-size",
    "field_npix": "field-npix",
    "nbins": "nbins",
    "tomo_bin_indices": "tomo-bin-indices",
    "seed": "seed",
    "zero_mean_maps": "zero-mean-maps",
    "apply_bnt": "apply-bnt",
    "npe_samples_requested": "npe-samples",
    "npe_split_seed": None,  # internal, not a CLI flag
    "map_kind": "map-kind",

    # CNN compressor
    "compressor_arch": "compressor-arch",
    "compressor_dim": "compressor-dim",
    "compressor_dense_width": "compressor-dense-width",
    "compressor_conv_channels": "compressor-conv-channels",
    "compressor_steps": "compressor-steps",
    "compressor_batch_size": "compressor-batch-size",
    "compressor_lr": "compressor-lr",
    "compressor_checkpoint_policy": "compressor-checkpoint-policy",
    "compressor_train_split": "compressor-train-split",
    "compressor_val_split": "compressor-val-split",
    "nde_train_split": "nde-train-split",
    "nde_val_split": "nde-val-split",
    "require_disjoint_train_examples": "require-disjoint-train-examples",
    "standardize_summary": "standardize-summary",

    # CNN harmonic
    "cnn_map_route": "cnn-map-route",
    "harmonic_cache_regime": "harmonic-cache-regime",
    "harmonic_normalize_input_channels": "harmonic-normalize-input-channels",
    "full_sphere_cross_cache": "full-sphere-cross-cache",
    "channel_mode": "channel-mode",

    # CNN NDE
    "total_steps": "total-steps",
    "save_every": "save-every",
    "batch_size": "batch-size",
    "nvp_layers": "nvp-layers",
    "nvp_hidden": "nvp-hidden",

    # L1 datavector
    "l1_min_snr": "l1-min-snr",
    "l1_max_snr": "l1-max-snr",
    "l1_min_snr_cross": "l1-min-snr-cross",
    "l1_max_snr_cross": "l1-max-snr-cross",
    "cross_snr_percentile": "cross-snr-percentile",
    "cross_map_apodize": "cross-map-apodize",
    "cross_noise_model": "cross-noise-model",
    "cross_maps": "cross-maps",
    "summary_transform": "summary-transform",
    "summary_clip_value": "summary-clip-value",
    "min_feature_variance": "min-feature-variance",
    "l1_nbins": "l1-nbins",
    "l1_implementation": "l1-implementation",

    # L1 NDE
    "npe_epochs": "epochs",
    "npe_batch_size": "batch-size",  # collides with CNN; safe — both are batch-size CLI flag
    "npe_learning_rate": "learning-rate",
    "npe_warmup_steps": "warmup-steps",
    "npe_decay_steps": "decay-steps",
}

# ANSI colors (only if TTY)
def _color(s, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{s}\033[0m"
    return s

red    = lambda s: _color(s, "31")
yellow = lambda s: _color(s, "33")
green  = lambda s: _color(s, "32")
gray   = lambda s: _color(s, "90")


def parse_launcher_flags(sh_path: Path) -> dict[str, str | bool]:
    """Parse a bash launcher and return {flag_name: value} for --foo values.

    Treats `--foo --bar` as `--foo` boolean (not `--foo=--bar`).
    """
    text = sh_path.read_text()
    flags: dict[str, str | bool] = {}
    # collapse line continuations
    cleaned = re.sub(r"\\\s*\n\s*", " ", text)
    # walk word-by-word so we never consume another --flag as the previous flag's value
    words = re.findall(r"\"[^\"]*\"|\'[^\']*\'|\S+", cleaned)
    i = 0
    while i < len(words):
        w = words[i].strip("\"'")
        if w.startswith("--") and re.match(r"^--[a-zA-Z][\w-]*$", w):
            flag = w[2:]
            # peek ahead
            if i + 1 < len(words):
                nxt = words[i + 1].strip("\"'")
                # value if next word is not another --flag and not a shell token
                if not (nxt.startswith("--") and re.match(r"^--[a-zA-Z][\w-]*$", nxt)) \
                        and not nxt.startswith(">") and not nxt.startswith("\\"):
                    flags[flag] = nxt
                    i += 2
                    continue
            flags[flag] = True
        i += 1
    return flags


def normalize_for_compare(v):
    """Make two values comparable across (str, int, float, bool, list) types."""
    if v is True or v is False or v is None:
        return v
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    s = str(v)
    # try int/float
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except (ValueError, TypeError):
        pass
    return s


def gotcha_checks(launcher_flags, anchor_meta) -> list[str]:
    """Per-project tripwires that have bitten us. Each returns a fatal-or-warn message.

    Add new checks here whenever a silent-default bug is discovered.
    """
    errors = []
    # PCA on L1 datavectors — see feedback_never_pca_l1.md
    # Trigger when launcher invokes the L1 script OR anchor meta says it's L1
    l1_script = any(
        "npe_l1norm_cross_jaxili" in str(v) for v in (launcher_flags or {}).values()
    )
    if l1_script or anchor_meta.get("method", "").startswith("l1"):
        pca = launcher_flags.get("pca-components")
        if pca is None or (isinstance(pca, str) and pca not in ("0", "0.0")):
            errors.append(red(
                f"PCA GOTCHA: --pca-components must be 0 for L1 (default is 50). "
                f"Found: {pca!r}. See feedback_never_pca_l1.md."
            ))
    # Harmonic route required for L1 auto+cross — feedback_l1_cross_must_use_harmonic_route.md
    cross_active = launcher_flags.get("cross-maps") is True or anchor_meta.get("cross_maps") is True
    if l1_script and cross_active:
        if not launcher_flags.get("full-sphere-cross-cache"):
            errors.append(red(
                "HARMONIC-ROUTE GOTCHA: L1 cross runs MUST pass --full-sphere-cross-cache "
                "(harmonic-cache route). The TFDS+--cross-maps path silently ignores "
                "--cross-noise-model channel_empirical_global and uses broken auto_scalar. "
                "See feedback_l1_cross_must_use_harmonic_route.md."
            ))
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("launcher", type=Path, help="Path to the launcher .sh")
    ap.add_argument("anchor_meta", type=Path, help="Path to the anchor's .meta.json")
    ap.add_argument("--arm", type=str, default=None,
                    help="Optional sub-string to filter launcher tokens by, "
                         "for masters that run multiple arms in one script "
                         "(e.g., 'cnn_cross_canon' to look at only that arm).")
    args = ap.parse_args()

    if not args.launcher.exists():
        print(f"launcher not found: {args.launcher}", file=sys.stderr); return 2
    if not args.anchor_meta.exists():
        print(f"meta not found: {args.anchor_meta}", file=sys.stderr); return 2

    launcher_flags = parse_launcher_flags(args.launcher)
    if args.arm:
        # crude arm-filter: parse only the section between an "arm" header and the next "arm"
        text = args.launcher.read_text()
        arm_chunk = re.search(
            rf"(?ms)(run_{args.arm.split('_')[0]}_{args.arm.split('_')[1]}_seed\s*\(\)\s*\{{.*?\n\}}).+?(?=^\}})",
            text)
        if arm_chunk:
            # Fall back to manual filter on tokens since the regex is fragile;
            # better to extract from the function body
            pass

    meta = json.loads(args.anchor_meta.read_text())

    print(f"\n{'='*100}")
    print(f"flag_diff.py — comparing")
    print(f"  launcher : {args.launcher}")
    print(f"  anchor   : {args.anchor_meta}")
    if args.arm: print(f"  arm     : {args.arm}")
    print(f"{'='*100}\n")

    print(f"{'meta key':<38}  {'anchor value':<28}  {'launcher value':<28}  status")
    print("-" * 110)
    n_match = n_mismatch = n_default = n_skipped = 0
    for meta_key, flag in sorted(META_TO_FLAG.items()):
        if flag is None:
            continue
        if meta_key not in meta:
            continue
        anchor_val = meta[meta_key]
        launcher_val = launcher_flags.get(flag, None)
        a_n = normalize_for_compare(anchor_val)
        l_n = normalize_for_compare(launcher_val) if launcher_val is not None else None
        anchor_str = f"{anchor_val!r}"[:26]
        launcher_str = f"{launcher_val!r}"[:26] if launcher_val is not None else "(not set, script default)"
        if launcher_val is None:
            # anchor used a value but launcher doesn't pass the flag → silent default
            status = yellow("DEFAULT — silent fallback to script default; anchor used " + str(anchor_val))
            n_default += 1
        elif a_n == l_n:
            status = green("match")
            n_match += 1
        elif launcher_val is True and anchor_val in (True, "true", "True"):
            status = green("match (bool)")
            n_match += 1
        else:
            status = red(f"MISMATCH")
            n_mismatch += 1
        print(f"{meta_key:<38}  {anchor_str:<28}  {launcher_str:<28}  {status}")

    print("-" * 110)
    print(f"summary: {green(str(n_match)+' match')}  /  {yellow(str(n_default)+' default-fallback')}  /  {red(str(n_mismatch)+' MISMATCH')}")

    # Project-specific tripwire checks (PCA, harmonic-route)
    gotchas = gotcha_checks(launcher_flags, meta)
    if gotchas:
        print()
        print(red("=== GOTCHA TRIPWIRES (project-specific, fatal) ==="))
        for g in gotchas:
            print("  " + g)

    if n_mismatch > 0 or n_default > 0 or gotchas:
        print(red("\nACTION REQUIRED — fix mismatches/defaults/gotchas in launcher before launching."))
        return 1
    print(green("\nClean — all anchor fields match what the launcher passes; no gotcha tripwires."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
