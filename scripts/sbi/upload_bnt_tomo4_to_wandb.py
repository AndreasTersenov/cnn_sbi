#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import wandb
from wandb.errors import UsageError


RUN_NAME_RE = re.compile(
    r"^(?P<method>cnn|l1|l1vmim)_tomo4_20deg160_(?P<condition>bnt|nobnt)_s(?P<seed>\d+)\.npy$"
)
INFLATION_KEY = "inflation_std_sum_bnt_over_nobnt"
EXPECTED_RUNS = 18


@dataclass(frozen=True)
class RunSpec:
    file_name: str
    run_name: str
    method: str
    condition: str
    seed: int
    std_sum: float
    bias_l2: float
    method_inflation_std_sum_bnt_over_nobnt: float
    contour_path: Path | None


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_specs(study_root: Path) -> List[RunSpec]:
    posterior_summary_path = study_root / "posterior_summary.json"
    bnt_metrics_summary_path = study_root / "bnt_metrics_summary.json"
    figures_dir = study_root / "figures"

    posterior_summary = _load_json(posterior_summary_path)
    bnt_metrics_summary = _load_json(bnt_metrics_summary_path)

    if not isinstance(posterior_summary, list):
        raise ValueError("posterior_summary.json must be a list")
    if not isinstance(bnt_metrics_summary, dict):
        raise ValueError("bnt_metrics_summary.json must be an object")

    inflation_by_method: Dict[str, float] = {}
    for method, block in bnt_metrics_summary.items():
        if not isinstance(block, dict):
            raise ValueError(f"bnt_metrics_summary['{method}'] must be an object")
        if INFLATION_KEY not in block:
            raise ValueError(f"Missing {INFLATION_KEY} for method '{method}'")
        inflation_by_method[method] = float(block[INFLATION_KEY])

    specs: List[RunSpec] = []
    seen = set()
    for row in posterior_summary:
        if not isinstance(row, dict):
            raise ValueError("posterior_summary.json entries must be objects")
        file_name = row.get("file")
        if not isinstance(file_name, str):
            raise ValueError("posterior_summary row missing 'file' string")
        match = RUN_NAME_RE.match(file_name)
        if not match:
            raise ValueError(f"Unexpected posterior filename pattern: {file_name}")

        method = match.group("method")
        condition = match.group("condition")
        seed = int(match.group("seed"))
        run_name = file_name[:-4]
        combo = (method, condition, seed)
        if combo in seen:
            raise ValueError(f"Duplicate run combination in posterior summary: {combo}")
        seen.add(combo)

        if method not in inflation_by_method:
            raise ValueError(f"Method '{method}' not found in bnt_metrics_summary.json")

        for metric in ("std_sum", "bias_l2"):
            if metric not in row:
                raise ValueError(f"Missing '{metric}' for {file_name}")

        contour_path = figures_dir / f"{run_name}.png"
        if not contour_path.exists():
            contour_path = None

        specs.append(
            RunSpec(
                file_name=file_name,
                run_name=run_name,
                method=method,
                condition=condition,
                seed=seed,
                std_sum=float(row["std_sum"]),
                bias_l2=float(row["bias_l2"]),
                method_inflation_std_sum_bnt_over_nobnt=float(inflation_by_method[method]),
                contour_path=contour_path,
            )
        )

    if len(specs) != EXPECTED_RUNS:
        raise ValueError(
            f"Expected {EXPECTED_RUNS} runs from posterior_summary.json, found {len(specs)}"
        )
    return sorted(specs, key=lambda s: (s.method, s.condition, s.seed))


def _run_config(spec: RunSpec) -> Dict[str, Any]:
    return {
        "method": spec.method,
        "condition": spec.condition,
        "seed": spec.seed,
        "study": "bnt_tomo4_study",
        "map_kind": "nbody",
        "field_size": 20,
        "field_npix": 160,
        "tomo_bins": "1,2,3,4",
        "n_scales": 5,
        "l1_bins": 40,
        "snr_range": "[-13,13]",
    }


def upload_runs(study_root: Path, specs: List[RunSpec]) -> Dict[str, Any]:
    uploaded_rows: List[Dict[str, Any]] = []
    project_url: str | None = None
    group_url: str | None = None

    for spec in specs:
        run = wandb.init(
            project="cnn_sbi_bnt",
            group="bnt_tomo4_study",
            name=spec.run_name,
            config=_run_config(spec),
            reinit=True,
            job_type="study_upload",
            tags=["bnt_tomo4_study", spec.method, spec.condition, f"seed{spec.seed}"],
        )
        try:
            metrics = {
                "std_sum": spec.std_sum,
                "bias_l2": spec.bias_l2,
                "method_inflation_std_sum_bnt_over_nobnt": spec.method_inflation_std_sum_bnt_over_nobnt,
            }
            wandb.log(metrics)
            if spec.contour_path is not None:
                wandb.log({"contour_image": wandb.Image(str(spec.contour_path))})

            run.summary.update(metrics)
            run.summary["posterior_file"] = spec.file_name
            if spec.contour_path is not None:
                run.summary["contour_image_file"] = str(spec.contour_path.relative_to(study_root))

            run_url = str(run.url)
            uploaded_rows.append(
                {
                    "run_name": spec.run_name,
                    "method": spec.method,
                    "condition": spec.condition,
                    "seed": spec.seed,
                    "url": run_url,
                }
            )

            if project_url is None and run.entity and run.project:
                project_url = f"https://wandb.ai/{run.entity}/{run.project}"
                group_url = f"{project_url}/groups/bnt_tomo4_study"
        finally:
            wandb.finish()

    if len(uploaded_rows) != EXPECTED_RUNS:
        raise RuntimeError(f"Uploaded {len(uploaded_rows)} runs, expected {EXPECTED_RUNS}")

    result = {
        "count": len(uploaded_rows),
        "project_url": project_url,
        "group_url": group_url,
        "runs": uploaded_rows,
    }
    return result


def main() -> int:
    study_root = (Path(__file__).resolve().parent / "bnt_tomo4_study").resolve()
    urls_path = study_root / "wandb_run_urls.json"
    try:
        specs = build_run_specs(study_root)
        result = upload_runs(study_root, specs)
    except (UsageError, wandb.errors.CommError) as exc:
        print(
            "W&B upload blocked: login/API key missing or network/access issue. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1

    urls_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Uploaded runs: {result['count']}")
    if result["project_url"]:
        print(f"Project: {result['project_url']}")
    if result["group_url"]:
        print(f"Group: {result['group_url']}")
    print("Run URLs:")
    for row in result["runs"]:
        print(row["url"])
    print(f"Saved run URL list: {urls_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
