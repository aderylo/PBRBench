"""Run a configured screen-space PBR material estimator."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_2d import PBREstimationDataset2D, PBREstimationSample2D  # noqa: E402
from src.methods_2d import BaseMaterialEstimator2D  # noqa: E402
from src.utils import get_pylogger  # noqa: E402

log = get_pylogger(__name__)


def get_pending_samples(
    dataset: Sequence[PBREstimationSample2D],
    predictions_dir: Path,
    *,
    overwrite: bool = False,
) -> list[PBREstimationSample2D]:
    """Scan prediction output directory and return pending samples."""
    return [
        sample
        for sample in dataset
        if overwrite or not (predictions_dir / sample.sample_id / ".SUCCESS").is_file()
    ]


def infer(config: DictConfig) -> None:
    """Instantiate dataset & estimator, filter pending samples, and run prediction."""
    log.info(f"Instantiating dataset <{config.data._target_}>")
    dataset: PBREstimationDataset2D = instantiate(config.data)

    log.info(f"Instantiating estimator <{config.method_2d._target_}>")
    estimator: BaseMaterialEstimator2D = instantiate(
        config.method_2d, project_root=PROJECT_ROOT
    )
    if not isinstance(estimator, BaseMaterialEstimator2D):
        raise TypeError(
            f"Expected BaseMaterialEstimator2D, got {type(estimator).__name__}"
        )

    output_dir = Path(config.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))

    overwrite = bool(config.get("runtime", {}).get("overwrite", False))
    pending = get_pending_samples(dataset, predictions_dir, overwrite=overwrite)
    log.info(f"{estimator.name}: {len(pending)} pending samples")
    if not pending:
        log.info("All samples are complete (.SUCCESS markers found). Exiting.")
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
