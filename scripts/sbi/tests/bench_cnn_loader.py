#!/usr/bin/env python
"""Benchmark harness for the CNN harmonic compressor data loaders.

Measures *integrated* training throughput (the real `npe_cnn_nbody_tomo.py
--train-compressor --exit-after-compress` step, not an isolated transfer
micro-benchmark — an isolated bench once hid a CpuDevice-placement bug) for
competing loader implementations, under identical config, repeated, with the
node/GPU contention state stamped on every run.

This exists because perf claims on this project have repeatedly been wrong when
assumed rather than measured (see HANDOFF_CNN_LOADER_REBUILD.md). The harness's
first job is to *reproduce the known baseline* (tfrecord ~14 it/s on a low-load
node with a free GPU). If it cannot, fix the harness before trusting any
candidate number.

Candidates
----------
  tfrecord  - current production path: tf.data + DLPack + thread-budget
              (`build_harmonic_tfrecord_iterator`). The number to beat.
  npz       - original GIL-bound zlib `.npz` loader (`build_harmonic_batch_iterator`).
              The slow floor (~2.4 it/s); sanity that the harness sees the gap.
  npz_raw   - (placeholder) the planned thin raw-bytes numpy loader. Wire its
              selecting flag into CANDIDATES once the loader exists.

What is measured (per run)
--------------------------
  - steady-state it/s: median + p10/p90 of tqdm instantaneous rates AFTER warmup
    (the final tqdm average is misleading — it folds in warmup + contention dips,
    which is exactly how "4.28 it/s avg" hid a "~14 then collapse to ~1" run).
  - GPU util (card-level) median/p90 over the measured window.
  - peak host threads and RSS of the training process.
  - startup -> first measured step latency.
Environment stamp (per run): load average, GPU co-tenants, CUDA_VISIBLE_DEVICES,
warm/cold note.

Usage
-----
  # validate the rate parser with no GPU (back-pressure check):
  python bench_cnn_loader.py --self-test

  # cheap plumbing smoke (number not meaningful):
  python bench_cnn_loader.py --candidate tfrecord --gpu 0 --smoke

  # real baseline reproduction (run in a LOW-LOAD window on a free-ish GPU):
  python bench_cnn_loader.py --candidate tfrecord --gpu 0 --steps 400 --runs 3 \
      --out results/diagnostics/bench_cnn_loader/tfrecord.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _kill_group(proc: "subprocess.Popen") -> None:
    """Kill the whole process group (conda run -> python -> grain workers)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.5)

# --------------------------------------------------------------------------- #
# Paths (verified 2026-05-29).
# --------------------------------------------------------------------------- #
SBI_DIR = Path(__file__).resolve().parent.parent
RUNNER = SBI_DIR / "npe_cnn_nbody_tomo.py"
HARM_NPZ_CACHE = (
    SBI_DIR / "results" / "exploratory" / "cross_maps_campaign" / "full_sphere_cache_grid"
)
TFRECORD_DIR = Path("/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid")
CONDA_ENV = "jaxili"

# Canonical heavy arm: 10-channel auto+cross, `plain` arch (lightest GPU step ->
# most loader-bound -> cleanest signal for comparing loaders), nobnt regime.
# Mirrors run_cross_only_campaign.py's Stage-A flags, with val+ckpt suppressed
# (save-every huge) so no validation pass contaminates train throughput.
BASE_FLAGS: List[str] = [
    "--map-kind", "nbody",
    "--tfds-name", "NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    "--field-size", "20", "--field-npix", "160",
    "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
    "--full-sphere-cross-cache", str(HARM_NPZ_CACHE),
    "--channel-mode", "auto_cross",
    "--no-wandb",
    "--train-compressor",
    "--compressor-save-every", "999999",
    "--compressor-batch-size", "128",
    "--compressor-dense-width", "256",
    "--compressor-train-split", "train",
    "--compressor-val-split", "val",
    "--nde-train-split", "train", "--nde-val-split", "val",
    "--cnn-map-route", "harmonic",
    "--harmonic-cache-regime", "nobnt",
    "--harmonic-normalize-input-channels",
    "--ds-batch-size", "500",
    "--compressor-arch", "plain",
    "--compressor-dim", "10",
    # NDE-budget flags: parsed but never used (--exit-after-compress returns first).
    "--total-steps", "2", "--batch-size", "16", "--patience", "0",
    "--save-every", "1", "--npe-samples", "64",
    "--exit-after-compress",
    "--seed", "41",
]

# ArrayRecord TFDS-cross dataset for the Grain candidates (subset for benchmarking).
GRAIN_TFDS_DIR = Path("/nas/tersenov/tfds_cross_arrayrecord_subset20")
# TFRecord TFDS-cross dataset for the standard-tf.data candidate (auto-only mechanism).
CROSS_TFRECORD_DIR = Path("/nas/tersenov/tfds_cross_tfrecord_subset20")
CROSS_TFRECORD_FULL_DIR = Path("/nas/tersenov/tfds_cross_tfrecord_full")

# Auto-only ANCHOR: the standard 4-channel tfds.load + tf.data path (the ~20 it/s
# reference). Distinct base command (no --full-sphere-cross-cache -> cnn_map_route=tfds),
# canonical plain-arm flags from autoresearch_cnn-auto-push/run_arm.py.
AUTO_BASE_FLAGS: List[str] = [
    "--map-kind", "nbody",
    "--tfds-name", "NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48",
    "--field-size", "20", "--field-npix", "160",
    "--nbins", "4", "--tomo-bin-indices", "1,2,3,4",
    "--zero-mean-maps",
    "--no-wandb",
    "--train-compressor",
    "--compressor-save-every", "999999",
    "--compressor-batch-size", "128",
    "--compressor-lr", "1e-3",
    "--compressor-dim", "16",
    "--compressor-dense-width", "512",
    "--compressor-conv-channels", "64,128,256",
    "--compressor-pool-window", "16",
    "--compressor-pool-stride", "8",
    "--compressor-arch", "plain",
    "--compressor-train-split", "train",
    "--compressor-val-split", "test",
    "--nde-train-split", "train", "--nde-val-split", "test",
    "--total-steps", "2", "--batch-size", "16", "--patience", "0",
    "--save-every", "1", "--npe-samples", "64",
    "--exit-after-compress",
    "--seed", "41",
]


def full_flags(label: str) -> List[str]:
    """Full npe_cnn flag list for a candidate (auto_tfds has its own base)."""
    if label == "auto_tfds":
        return list(AUTO_BASE_FLAGS)
    return list(BASE_FLAGS) + CANDIDATES[label]


# label -> extra flags that SELECT the loader. Everything else is held identical.
CANDIDATES: Dict[str, List[str]] = {
    "tfrecord": ["--harmonic-tfrecord-dir", str(TFRECORD_DIR)],
    "npz": [],  # no tfrecord dir -> build_harmonic_batch_iterator (.npz/zlib)
    # Grain on the ArrayRecord TFDS dataset (worker processes via mp_prefetch).
    "grain_w8": ["--grain-tfds-dir", str(GRAIN_TFDS_DIR), "--grain-num-workers", "8"],
    "grain_w16": ["--grain-tfds-dir", str(GRAIN_TFDS_DIR), "--grain-num-workers", "16"],
    "grain_w32": ["--grain-tfds-dir", str(GRAIN_TFDS_DIR), "--grain-num-workers", "32"],
    # Standard tfds.load + tf.data on a TFRecord cross dataset (mirrors auto-only).
    "tfdata_cross": ["--cross-tfdata-dir", str(CROSS_TFRECORD_DIR)],
    "tfdata_cross_full": ["--cross-tfdata-dir", str(CROSS_TFRECORD_FULL_DIR)],
    # Channel-mode sanity variants (later flags override BASE_FLAGS' --channel-mode auto_cross).
    "tfdata_cross_full_cronly": ["--cross-tfdata-dir", str(CROSS_TFRECORD_FULL_DIR), "--channel-mode", "cross_only"],
    "tfdata_cross_full_autonly": ["--cross-tfdata-dir", str(CROSS_TFRECORD_FULL_DIR), "--channel-mode", "auto_only"],
}

# Selectable candidates incl. the auto-only anchor (its own base via full_flags).
ALL_CANDIDATES: List[str] = list(CANDIDATES) + ["auto_tfds"]

# --------------------------------------------------------------------------- #
# tqdm rate parsing.
# --------------------------------------------------------------------------- #
# Matches e.g. "404/500 [00:44<00:06, 15.54it/s]" and "499/500 [01:55<00:01, 1.04s/it]"
_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?,\s*([\d.]+)(it/s|s/it)\]")


def parse_tqdm_rates(text: str) -> List[Tuple[int, float]]:
    """Return [(step, it_per_s), ...] for every tqdm update in `text`.

    `s/it` values are inverted to it/s so a single distribution is comparable.
    """
    out: List[Tuple[int, float]] = []
    for m in _TQDM_RE.finditer(text):
        step = int(m.group(1))
        val = float(m.group(3))
        unit = m.group(4)
        if val <= 0:
            continue
        rate = val if unit == "it/s" else 1.0 / val
        out.append((step, rate))
    return out


def summarize_rates(rates: List[Tuple[int, float]], warmup: int) -> Dict:
    """Drop steps <= warmup, summarize the rest. Median is the headline; p10/p90
    capture the contention spread that a single average hides."""
    kept = [r for (s, r) in rates if s > warmup]
    if not kept:
        return {"n": 0}
    kept_sorted = sorted(kept)

    def pct(p: float) -> float:
        if len(kept_sorted) == 1:
            return kept_sorted[0]
        idx = min(len(kept_sorted) - 1, max(0, int(round(p * (len(kept_sorted) - 1)))))
        return kept_sorted[idx]

    return {
        "n": len(kept),
        "median_it_s": round(statistics.median(kept), 3),
        "p10_it_s": round(pct(0.10), 3),
        "p90_it_s": round(pct(0.90), 3),
        "min_it_s": round(min(kept), 3),
        "max_it_s": round(max(kept), 3),
    }


# --------------------------------------------------------------------------- #
# Environment / process sampling.
# --------------------------------------------------------------------------- #
def load_avg() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return float("nan")


def gpu_compute_apps(gpu: int) -> List[str]:
    """Co-tenant processes on the physical GPU index (so the reader knows the
    card was not exclusive)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu),
             "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def gpu_util(gpu: int) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu),
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


def find_child_pid(unique_token: str) -> Optional[int]:
    """The python training process matching `unique_token` (our unique save-dir),
    not the `conda run` wrapper. Among matches, prefer cmdline[0] == python*."""
    candidates: List[int] = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if unique_token in cmd and "npe_cnn_nbody_tomo.py" in cmd:
            first = cmd.split()[0] if cmd.split() else ""
            if Path(first).name.startswith("python"):
                candidates.append(int(p.name))
    return max(candidates) if candidates else None


def proc_threads_rss(pid: int) -> Tuple[Optional[int], Optional[float]]:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except Exception:
        return None, None
    threads = rss_mb = None
    for line in status.splitlines():
        if line.startswith("Threads:"):
            threads = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            rss_mb = float(line.split()[1]) / 1024.0
    return threads, rss_mb


class Sampler(threading.Thread):
    """Polls (every `interval`s) the training process's threads/RSS and the
    card-level GPU util while the run is alive."""

    def __init__(self, gpu: int, unique_token: str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.unique_token = unique_token
        self.interval = interval
        self._stop_event = threading.Event()
        self.samples: List[Dict] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            t = time.time()
            pid = find_child_pid(self.unique_token)
            threads, rss = (None, None)
            if pid is not None:
                threads, rss = proc_threads_rss(pid)
            self.samples.append(
                {"t": t, "pid": pid, "threads": threads, "rss_mb": rss,
                 "gpu_util": gpu_util(self.gpu)}
            )
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def summarize(self, skip_seconds: float) -> Dict:
        if not self.samples:
            return {}
        t0 = self.samples[0]["t"]
        late = [s for s in self.samples if s["t"] - t0 >= skip_seconds]
        threads = [s["threads"] for s in self.samples if s["threads"]]
        rss = [s["rss_mb"] for s in self.samples if s["rss_mb"]]
        util = [s["gpu_util"] for s in late if s["gpu_util"] == s["gpu_util"]]  # drop nan
        out: Dict = {"n_samples": len(self.samples)}
        if threads:
            out["max_threads"] = max(threads)
            out["median_threads"] = int(statistics.median(threads))
        if rss:
            out["peak_rss_gb"] = round(max(rss) / 1024.0, 2)
        if util:
            out["gpu_util_median"] = round(statistics.median(util), 1)
            out["gpu_util_p90"] = round(sorted(util)[min(len(util) - 1, int(0.9 * (len(util) - 1)))], 1)
        return out


# --------------------------------------------------------------------------- #
# One integrated run.
# --------------------------------------------------------------------------- #
def run_once(label: str, gpu: int, steps: int, warmup: int, mem_fraction: float,
             out_root: Path, run_idx: int) -> Dict:
    if label not in ALL_CANDIDATES:
        raise SystemExit(f"unknown candidate {label!r}; choices: {ALL_CANDIDATES}")

    run_dir = out_root / f"{label}_gpu{gpu}_run{run_idx}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    unique_token = str(run_dir)  # uniquely identifies this child in /proc
    log_path = run_dir / "stdout.log"

    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python", "-u",
        str(RUNNER),
        *full_flags(label),
        "--compressor-steps", str(steps),
        "--save-dir", str(run_dir),
        "--cache-dir", str(run_dir / "cache"),
        "--posterior-out", str(run_dir / "unused.npy"),
        "--figure-out", str(run_dir / "unused.pdf"),
        "--cuda-visible-devices", str(gpu),
    ]

    env = dict(os.environ)
    # Be a polite co-tenant on a shared card; plain CNN needs little VRAM.
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    # Deliberately do NOT touch OMP_* — the script sets its own CNN_CPU_THREADS
    # budget; we measure that real production thread behavior.

    pre = {"load": load_avg(), "co_tenants": gpu_compute_apps(gpu),
           "cuda_visible_devices": str(gpu)}

    sampler = Sampler(gpu=gpu, unique_token=unique_token)
    t_start = time.time()
    print(f"  [{label} run{run_idx}] launching on GPU {gpu}, steps={steps} "
          f"(load={pre['load']}, co-tenants={len(pre['co_tenants'])}) ...")
    status = "completed"
    with open(log_path, "w") as logf:
        # start_new_session so we can kill the whole tree (conda run -> python -> grain workers).
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        sampler.start()
        # We only want TRAINING throughput. With --exit-after-compress the child runs the
        # slow compress_dataset (.npz NDE-prep) AFTER training; kill at the training-done
        # marker so the benchmark measures training only and stays fast.
        timeout = steps * 3.0 + 1200.0
        deadline = t_start + timeout
        while True:
            if proc.poll() is not None:
                break
            if time.time() > deadline:
                _kill_group(proc)
                status = "timeout"
                print(f"  [{label} run{run_idx}] TIMEOUT after {timeout:.0f}s — killed.")
                break
            if "Compressor training done" in log_path.read_text(errors="ignore"):
                time.sleep(1.5)  # let the final tqdm line flush
                _kill_group(proc)
                status = "killed_after_train"
                break
            time.sleep(2.0)
    sampler.stop()
    sampler.join(timeout=5)
    wall = time.time() - t_start

    text = log_path.read_text(errors="ignore")
    rates = parse_tqdm_rates(text)
    summary = summarize_rates(rates, warmup)
    proc_summary = sampler.summarize(skip_seconds=max(15.0, warmup * 0.3))
    post = {"load": load_avg()}

    result = {
        "candidate": label, "gpu": gpu, "run_idx": run_idx, "steps": steps,
        "warmup": warmup, "wall_s": round(wall, 1), "exit_code": proc.returncode,
        "status": status,
        "n_tqdm_updates": len(rates),
        "throughput": summary, "process": proc_summary,
        "env_pre": pre, "env_post": post, "log": str(log_path),
    }
    if summary.get("n"):
        print(f"  [{label} run{run_idx}] median {summary['median_it_s']} it/s "
              f"(p10 {summary['p10_it_s']} / p90 {summary['p90_it_s']}), "
              f"GPU util ~{proc_summary.get('gpu_util_median', '?')}%, "
              f"threads max {proc_summary.get('max_threads', '?')}, "
              f"exit={proc.returncode}, load {pre['load']}->{post['load']}")
    else:
        print(f"  [{label} run{run_idx}] NO tqdm rates parsed "
              f"(exit={proc.returncode}). Inspect {log_path}")
    return result


# --------------------------------------------------------------------------- #
# Aggregation + report.
# --------------------------------------------------------------------------- #
def aggregate(runs: List[Dict]) -> Dict:
    meds = [r["throughput"]["median_it_s"] for r in runs if r["throughput"].get("n")]
    agg: Dict = {"runs": len(runs), "runs_with_data": len(meds)}
    if meds:
        agg["median_of_run_medians_it_s"] = round(statistics.median(meds), 3)
        agg["min_run_median_it_s"] = round(min(meds), 3)
        agg["max_run_median_it_s"] = round(max(meds), 3)
    return agg


def write_report(out_path: Path, candidate: str, runs: List[Dict], agg: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# CNN loader benchmark — candidate `{candidate}`",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · runner `{RUNNER.name}` · "
        f"arm: 10-ch auto+cross, `plain`, nobnt.",
        "",
        "**Methodology:** integrated `--train-compressor --exit-after-compress` run; "
        "steady-state it/s = median of tqdm rates after warmup (NOT the tqdm average). "
        "Each run stamps load avg + GPU co-tenants. See `HANDOFF_CNN_LOADER_REBUILD.md`.",
        "",
        "| run | median it/s | p10 | p90 | n | GPU util% | max thr | peak RSS GB | load pre→post | co-ten | exit |",
        "|----:|------------:|----:|----:|--:|----------:|--------:|------------:|:-------------:|:------:|:----:|",
    ]
    for r in runs:
        t = r["throughput"]
        p = r["process"]
        med = t.get("median_it_s", "—")
        lines.append(
            f"| {r['run_idx']} | {med} | {t.get('p10_it_s','—')} | {t.get('p90_it_s','—')} "
            f"| {t.get('n','0')} | {p.get('gpu_util_median','?')} | {p.get('max_threads','?')} "
            f"| {p.get('peak_rss_gb','?')} | {r['env_pre']['load']}→{r['env_post']['load']} "
            f"| {len(r['env_pre']['co_tenants'])} | {r['exit_code']} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- runs: {agg['runs']} (with data: {agg['runs_with_data']})",
    ]
    if "median_of_run_medians_it_s" in agg:
        lines.append(
            f"- **median of run-medians: {agg['median_of_run_medians_it_s']} it/s** "
            f"(range {agg['min_run_median_it_s']}–{agg['max_run_median_it_s']})"
        )
    lines += [
        "",
        "> Interpret with the load column. A high median at low load is the free-CPU "
        "number; a low median with high load is contention, not the loader. To compare "
        "candidates, line up runs at *similar* load — or run them back-to-back in the "
        "same window.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    out_path.with_suffix(".json").write_text(json.dumps({"aggregate": agg, "runs": runs}, indent=2))
    print(f"\nWrote {out_path}\nWrote {out_path.with_suffix('.json')}")


# --------------------------------------------------------------------------- #
# Self-test (no GPU): validates the rate parser, the trickiest piece.
# --------------------------------------------------------------------------- #
_SELFTEST_LOG = (
    "Compressor[t]:  80%|####  | 400/500 [00:44<00:07, 14.06it/s]"
    "Compressor[t]:  81%|####  | 404/500 [00:44<00:06, 15.54it/s]"
    "Compressor[t]:  82%|####  | 409/500 [00:46<00:30,  2.99it/s]"
    "Compressor[t]: 100%|######| 499/500 [01:55<00:01,  1.04s/it]"  # s/it -> 0.96 it/s
)


def self_test() -> int:
    rates = parse_tqdm_rates(_SELFTEST_LOG)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        ok = ok and cond

    check("parsed 4 updates", len(rates) == 4, f"got {len(rates)}")
    steps = [s for s, _ in rates]
    check("steps in order", steps == [400, 404, 409, 499], f"got {steps}")
    check("it/s parsed", abs(rates[1][1] - 15.54) < 1e-6, f"got {rates[1][1]}")
    check("s/it inverted", abs(rates[3][1] - 1.0 / 1.04) < 1e-6, f"got {rates[3][1]}")
    # warmup drop: keep steps > 405 -> only 409 (2.99) and 499 (0.96)
    summ = summarize_rates(rates, warmup=405)
    check("warmup drop keeps 2", summ["n"] == 2, f"got {summ.get('n')}")
    check("median captures dip", abs(summ["median_it_s"] - statistics.median([2.99, 1.0 / 1.04])) < 1e-3,
          f"got {summ['median_it_s']}")
    # full (warmup 0) median should NOT equal the misleading low average
    summ_all = summarize_rates(rates, warmup=0)
    check("median != naive avg", summ_all["median_it_s"] > 2.99,
          f"median {summ_all['median_it_s']} should sit among healthy rates")
    print(f"\nself-test: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", choices=ALL_CANDIDATES, default="tfrecord")
    ap.add_argument("--gpu", type=int, default=0,
                    help="physical GPU index (project rule: GPU 1 for new jobs; "
                         "GPU 0 OK if <=45%%; never disrupt the L1 campaign).")
    ap.add_argument("--steps", type=int, default=400,
                    help="compressor steps; ~400 gives a steady window past warmup.")
    ap.add_argument("--warmup", type=int, default=40,
                    help="drop tqdm rates at step <= this (JIT + ramp).")
    ap.add_argument("--runs", type=int, default=3, help="repeat count (>=3 advised).")
    ap.add_argument("--mem-fraction", type=float, default=0.3,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION (polite on a shared card).")
    ap.add_argument("--out", type=str, default=None,
                    help="markdown report path; default under results/diagnostics/.")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the tqdm parser with no GPU, then exit.")
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny run (steps=30) to validate plumbing; number not meaningful.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    out_root = SBI_DIR / "results" / "diagnostics" / "bench_cnn_loader"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps, args.runs = 30, 1
        print("SMOKE: steps=30, runs=1 — validates plumbing only, it/s NOT meaningful.")

    print(f"Benchmarking candidate '{args.candidate}' on GPU {args.gpu}: "
          f"{args.runs} run(s) x {args.steps} steps (warmup {args.warmup}).")
    runs = [run_once(args.candidate, args.gpu, args.steps, args.warmup,
                     args.mem_fraction, out_root, i) for i in range(args.runs)]
    agg = aggregate(runs)

    out_path = Path(args.out) if args.out else out_root / f"{args.candidate}_gpu{args.gpu}.md"
    write_report(out_path, args.candidate, runs, agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
