"""Script to compute summary statistics, LaTeX table rows, and aggregate multi-method top/worst/representative sample candidates for direct PBR intrinsics evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from collections import defaultdict

import numpy as np
import rootutils

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

METHOD_NAMES = {
    "supermat": "SuperMat",
    "diffusion_renderer": "DiffusionRenderer",
    "neural_lightrig": "Neural LightRig",
    "pbr_oracle": "PBROracleSV (Ours)",
}

DATASETS = ["dtc", "objaverse", "polyhaven", "texverse"]


def parse_args():
    parser = argparse.ArgumentParser(description="Report direct PBR intrinsics evaluation statistics.")
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pbr_2d",
        help="Path to 2D predictions root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "report_intrinsics.yaml",
        help="Path to save YAML report.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    pbr_dir = args.predictions_dir
    report_data = {"methods": {}, "sample_highlights": {}}

    method_samples = {}
    sample_scores = defaultdict(dict)

    for method_id, display_name in METHOD_NAMES.items():
        metrics_path = pbr_dir / method_id / "metrics_direct.yaml"
        if not metrics_path.exists():
            metrics_path = pbr_dir / method_id / "metrics.yaml"
        if not metrics_path.exists():
            print(f"Skipping {method_id}: {metrics_path} not found.")
            continue

        data = load_yaml(metrics_path)
        samples = data.get("samples", {})
        if not samples:
            continue
        method_samples[method_id] = samples

        ds_samples = {ds: [] for ds in DATASETS}
        for s_id, s_data in samples.items():
            src = s_data.get("source", "")
            if not src:
                src = s_id.split("__")[0]
            if src in ds_samples:
                ds_samples[src].append((s_id, s_data))

            albedo_psnr = s_data.get("metrics", {}).get("albedo", {}).get("psnr")
            if albedo_psnr is not None:
                sample_scores[s_id][method_id] = albedo_psnr

        method_summary = {}
        for ds, s_list in ds_samples.items():
            if not s_list:
                continue

            channels = ["albedo", "roughness", "metallic"]
            ds_metrics = {}
            for ch in channels:
                psnrs = [item[1]["metrics"][ch]["psnr"] for item in s_list if ch in item[1]["metrics"] and "psnr" in item[1]["metrics"][ch]]
                rmses = [item[1]["metrics"][ch]["rmse"] for item in s_list if ch in item[1]["metrics"] and "rmse" in item[1]["metrics"][ch]]
                ssims = [item[1]["metrics"][ch]["ssim"] for item in s_list if ch in item[1]["metrics"] and "ssim" in item[1]["metrics"][ch]]
                lpipss = [item[1]["metrics"][ch]["lpips"] for item in s_list if ch in item[1]["metrics"] and "lpips" in item[1]["metrics"][ch]]

                ds_metrics[ch] = {
                    "psnr": float(np.mean(psnrs)) if psnrs else None,
                    "rmse": float(np.mean(rmses)) if rmses else None,
                    "ssim": float(np.mean(ssims)) if ssims else None,
                    "lpips": float(np.mean(lpipss)) if lpipss else None,
                }
            method_summary[ds] = ds_metrics

        report_data["methods"][method_id] = {
            "display_name": display_name,
            "metrics": method_summary,
        }

    agg_scores = []
    for s_id, scores in sample_scores.items():
        if len(scores) >= len(method_samples):
            avg_psnr = float(np.mean(list(scores.values())))
            agg_scores.append((s_id, avg_psnr, scores))

    agg_scores.sort(key=lambda x: x[1], reverse=True)
    n = len(agg_scores)

    best_20 = agg_scores[:20]
    mid_start = max(0, n // 2 - 10)
    repr_20 = agg_scores[mid_start : mid_start + 20]
    worst_20 = agg_scores[-20:]

    def format_highlights(items):
        res = []
        for s_id, avg_p, scores in items:
            res.append({
                "sample_id": s_id,
                "aggregate_albedo_psnr": round(avg_p, 3),
                "per_method_psnr": {m: round(score, 3) for m, score in scores.items()},
            })
        return res

    report_data["sample_highlights"] = {
        "best": format_highlights(best_20),
        "representative": format_highlights(repr_20),
        "worst": format_highlights(worst_20),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)

    print(f"Intrinsics Report saved to: {args.output}")


if __name__ == "__main__":
    main()
