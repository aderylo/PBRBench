"""Hydra script to generate direct PBR intrinsics material map visualization figures for Easy, Medium, and Hard sample splits with distinct 3D objects."""

from __future__ import annotations

from pathlib import Path
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import hydra
import rootutils
from omegaconf import DictConfig

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_tight_bbox(mask: np.ndarray, margin_pct: float = 0.03) -> tuple[int, int, int, int]:
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]

    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()

    h, w = ymax - ymin, xmax - xmin
    margin_y = int(h * margin_pct) + 3
    margin_x = int(w * margin_pct) + 3

    ymin = max(0, ymin - margin_y)
    ymax = min(mask.shape[0], ymax + margin_y)
    xmin = max(0, xmin - margin_x)
    xmax = min(mask.shape[1], xmax + margin_x)

    crop_h = ymax - ymin
    crop_w = xmax - xmin
    max_dim = max(crop_h, crop_w)

    cy = (ymin + ymax) // 2
    cx = (xmin + xmax) // 2

    ymin_sq = max(0, cy - max_dim // 2)
    ymax_sq = min(mask.shape[0], ymin_sq + max_dim)
    xmin_sq = max(0, cx - max_dim // 2)
    xmax_sq = min(mask.shape[1], xmin_sq + max_dim)

    return xmin_sq, ymin_sq, xmax_sq, ymax_sq


def load_masked_cropped_image(
    image_path: Path,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg_color: tuple[int, int, int] = (255, 255, 255),
    add_outline: bool = False,
) -> np.ndarray:
    h_mask, w_mask = mask.shape
    if not image_path.exists():
        img_arr = np.full((h_mask, w_mask, 3), bg_color[0], dtype=np.uint8)
    else:
        raw_img = Image.open(image_path)
        img_rgb = raw_img.convert("RGB")
        rgb = np.array(img_rgb)
        img_arr = np.full_like(rgb, bg_color[0], dtype=np.uint8)
        img_arr[mask] = rgb[mask]

    if add_outline:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        edges = np.array(mask_img.filter(ImageFilter.FIND_EDGES)) > 40
        img_arr[edges] = [130, 130, 130]

    return img_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]]


def render_intrinsics_figure(samples: list[dict], methods: list, data_dir: Path, predictions_dir: Path, output_pdf: Path, output_png: Path):
    n_cols = len(samples)
    n_rows = len(methods) + 1

    fig = plt.figure(figsize=(4.0 * n_cols, 3.2 * n_rows), dpi=300)
    gs_main = gridspec.GridSpec(
        n_rows,
        n_cols,
        height_ratios=[1.0] * n_rows,
        hspace=0.20,
        wspace=0.22,
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.02,
    )

    for col_idx, item in enumerate(samples):
        col_x = 0.08 + (col_idx + 0.5) * (0.90 / n_cols)
        hdr_title = item.get("title", f"{item['dataset']}/{item['object_id']}")
        fig.text(col_x, 0.965, hdr_title, fontsize=15, fontweight="bold", ha="center", va="center")

    fig.text(0.02, 0.88, "Inputs", fontsize=15, fontweight="bold", ha="center", va="center", rotation=90)
    for col_idx, item in enumerate(samples):
        ds, obj, view, in_l = item["dataset"], item["object_id"], item["view_id"], item["in_light"]
        eval_dir = data_dir / ds / obj / view
        mask_p = eval_dir / "mask.png"
        mask = (np.array(Image.open(mask_p).convert("L")) > 128) if mask_p.exists() else np.ones((512, 512), dtype=bool)
        bbox = get_tight_bbox(mask)

        rgb_path = eval_dir / "rgb" / f"{in_l}.png"
        input_img = load_masked_cropped_image(rgb_path, mask, bbox, add_outline=False)

        ax = fig.add_subplot(gs_main[0, col_idx])
        ax.imshow(input_img)
        ax.axis("off")

    for m_idx, m_info in enumerate(methods):
        row_idx = m_idx + 1
        m_id = m_info["id"] if isinstance(m_info, dict) else m_info.id
        m_name = m_info["name"] if isinstance(m_info, dict) else m_info.name

        row_y = 0.83 - (m_idx + 0.5) * (0.80 / len(methods))
        fig.text(0.02, row_y, m_name, fontsize=15, fontweight="bold", ha="center", va="center", rotation=90)

        for col_idx, item in enumerate(samples):
            ds, obj, view, in_l = item["dataset"], item["object_id"], item["view_id"], item["in_light"]
            sample_key = f"{ds}__{obj}__{view}__{in_l}"
            eval_dir = data_dir / ds / obj / view
            mask_p = eval_dir / "mask.png"
            mask = (np.array(Image.open(mask_p).convert("L")) > 128) if mask_p.exists() else np.ones((512, 512), dtype=bool)
            bbox = get_tight_bbox(mask)

            if m_id == "gt":
                alb_p = eval_dir / "albedo.png"
                rgh_p = eval_dir / "roughness.png"
                met_p = eval_dir / "metallic.png"
            else:
                pred_dir = predictions_dir / m_id / "predictions" / sample_key
                alb_p = pred_dir / "albedo.png"
                rgh_p = pred_dir / "roughness.png"
                met_p = pred_dir / "metallic.png"

            img_alb = load_masked_cropped_image(alb_p, mask, bbox, add_outline=False)
            img_rgh = load_masked_cropped_image(rgh_p, mask, bbox, add_outline=True)
            img_met = load_masked_cropped_image(met_p, mask, bbox, add_outline=True)

            cell_gs = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=gs_main[row_idx, col_idx],
                width_ratios=[1.0, 0.48],
                height_ratios=[1.0, 1.0],
                wspace=0.06,
                hspace=0.06,
            )

            ax_alb = fig.add_subplot(cell_gs[:, 0])
            ax_alb.imshow(img_alb)
            ax_alb.axis("off")

            ax_rgh = fig.add_subplot(cell_gs[0, 1])
            ax_rgh.imshow(img_rgh)
            ax_rgh.axis("off")

            ax_met = fig.add_subplot(cell_gs[1, 1])
            ax_met.imshow(img_met)
            ax_met.axis("off")

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated Intrinsics Visual: {output_pdf}")


@hydra.main(
    version_base="1.3",
    config_path="../configs/scripts",
    config_name="visuals_intrinsics",
)
def main(cfg: DictConfig) -> None:
    predictions_dir = project_path(cfg.predictions_dir)
    data_dir = project_path(cfg.data_dir)
    output_dir = project_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = cfg.methods

    report_path = PROJECT_ROOT / "outputs" / "reports" / "report_intrinsics.yaml"
    splits = {"easy": "best", "medium": "representative", "hard": "worst"}

    if report_path.exists():
        with report_path.open("r") as f:
            report = yaml.safe_load(f)

        for split_name, highlight_key in splits.items():
            highlight_list = report.get("sample_highlights", {}).get(highlight_key, [])
            if not highlight_list:
                continue

            seen_objects = set()
            samples = []
            for item in highlight_list:
                k = item["sample_id"]
                parts = k.split("__")
                ds, obj, view, in_l = parts[0], parts[1], parts[2], parts[3]
                if obj in seen_objects:
                    continue
                seen_objects.add(obj)
                samples.append({
                    "dataset": ds,
                    "object_id": obj,
                    "view_id": view,
                    "in_light": in_l,
                    "title": f"{obj[:15]} ({ds.capitalize()})",
                })
                if len(samples) == 4:
                    break

            out_pdf = output_dir / f"visuals_intrinsics_{split_name}.pdf"
            out_png = output_dir / f"visuals_intrinsics_{split_name}.png"
            try:
                render_intrinsics_figure(samples, methods, data_dir, predictions_dir, out_pdf, out_png)
            except Exception as e:
                print(f"Error generating {out_pdf}: {e}")
    else:
        samples = [dict(s) for s in cfg.samples]
        out_pdf = output_dir / cfg.output_filename
        out_png = output_dir / cfg.output_filename.replace(".pdf", ".png")
        render_intrinsics_figure(samples, methods, data_dir, predictions_dir, out_pdf, out_png)


if __name__ == "__main__":
    main()
