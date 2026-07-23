"""Run a configured screen-space PBR material estimator."""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_2d import PBREstimationDataset2D  # noqa: E402
from src.methods_2d import BaseMaterialEstimator2D  # noqa: E402
from src.utils import get_pylogger  # noqa: E402

log = get_pylogger(__name__)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def is_complete(sample_dir: Path) -> bool:
    """Check whether a sample prediction directory contains required material channel outputs."""
    required_channels = ("albedo.png", "roughness.png", "metallic.png")
    return sample_dir.is_dir() and all(
        (sample_dir / filename).is_file() for filename in required_channels
    )


def infer(config: DictConfig) -> None:
    """Instantiate dataset & estimator, filter pending samples, and run prediction."""
    log.info(f"Instantiating dataset <{config.data._target_}>")
    dataset_overrides = {}
    if hasattr(config.data, "root") and config.data.root:
        dataset_overrides["root"] = project_path(config.data.root)
    dataset = instantiate(config.data, **dataset_overrides)

    log.info(f"Instantiating estimator <{config.method_2d._target_}>")
    estimator: BaseMaterialEstimator2D = instantiate(
        config.method_2d, project_root=PROJECT_ROOT
    )
    if not isinstance(estimator, BaseMaterialEstimator2D):
        raise TypeError(
            f"Expected BaseMaterialEstimator2D, got {type(estimator).__name__}"
        )

    output_dir = project_path(config.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))

    samples = list(dataset)
    overwrite = (
        bool(config.runtime.get("overwrite", False))
        if hasattr(config, "runtime") and config.runtime
        else False
    )
    pending = [
        sample
        for sample in samples
        if overwrite or not is_complete(predictions_dir / sample.sample_id)
    ]

    log.info(
        f"{estimator.name}: {len(pending)} pending, "
        f"{len(samples) - len(pending)} complete, {len(samples)} requested"
    )
    if not pending:
        log.info("All samples are complete. Exiting.")
        return

    started = time.time()
    estimator.setup()
    try:
        estimator.predict(pending, predictions_dir)
        elapsed = time.time() - started
        log.info(
            f"Completed {len(pending)} predictions with {estimator.name} "
            f"in {elapsed:.2f}s ({elapsed / len(pending):.2f}s/sample)"
        )
    finally:
        estimator.teardown()


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer_pbr_2d")
def main(config: DictConfig) -> None:
    infer(config)


if __name__ == "__main__":
    main()
