"""Script to compute summary statistics, LaTeX table rows, and aggregate multi-method top/worst/representative sample candidates for indirect PBR relighting evaluation."""

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
}

DATASETS = ["dtc", "objaverse", "polyhaven", "texverse"]


def parse_args():
    parser = argparse.ArgumentParser(description="Report indirect PBR relighting evaluation statistics.")
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pbr_2d",
        help="Path to 2D predictions root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "report_rerendering.yaml",
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
        metrics_path = pbr_dir / method_id / "metrics_indirect.yaml"
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

            in_light = s_data.get("light_id", "")
            targets = s_data.get("targets", {})
            psnrs = [m["psnr"] for tgt_id, m in targets.items() if tgt_id != in_light]
            if psnrs:
                sample_scores[s_id][method_id] = float(np.mean(psnrs))

        method_summary = {}
        for ds, s_list in ds_samples.items():
            if not s_list:
                continue

            relight_psnrs, relight_ssims, relight_lpipss = [], [], []
            cycle_psnrs, cycle_ssims, cycle_lpipss = [], [], []

            for s_id, s_data in s_list:
                in_light = s_data.get("light_id", "")
                targets = s_data.get("targets", {})
                for tgt_id, m in targets.items():
                    if tgt_id == in_light:
                        cycle_psnrs.append(m["psnr"])
                        cycle_ssims.append(m["ssim"])
                        cycle_lpipss.append(m["lpips"])
                    else:
                        relight_psnrs.append(m["psnr"])
                        relight_ssims.append(m["ssim"])
                        relight_lpipss.append(m["lpips"])

            method_summary[ds] = {
                "relighting": {
                    "psnr": float(np.mean(relight_psnrs)) if relight_psnrs else None,
                    "ssim": float(np.mean(relight_ssims)) if relight_ssims else None,
                    "lpips": float(np.mean(relight_lpipss)) if relight_lpipss else None,
                },
                "cycle_consistency": {
                    "psnr": float(np.mean(cycle_psnrs)) if cycle_psnrs else None,
                    "ssim": float(np.mean(cycle_ssims)) if cycle_ssims else None,
                    "lpips": float(np.mean(cycle_lpipss)) if cycle_lpipss else None,
                },
            }

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
                "aggregate_relighting_psnr": round(avg_p, 3),
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

    print(f"Rerendering Report saved to: {args.output}")


if __name__ == "__main__":
    main()
