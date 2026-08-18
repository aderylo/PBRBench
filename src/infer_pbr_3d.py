"""Run a configured 3D PBR material estimator."""

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

from src.data.pbr_estimation_dataset_3d import PBREstimationDataset3D, PBREstimationSample3D  # noqa: E402
from src.methods_3d import BaseMaterialEstimator3D, Prediction3D  # noqa: E402
from src.utils import get_pylogger  # noqa: E402

log = get_pylogger(__name__)


def get_pending_samples(
    dataset: Sequence[PBREstimationSample3D],
    predictions_dir: Path,
    *,
    overwrite: bool = False,
) -> list[PBREstimationSample3D]:
    """Scan prediction output directory and return pending samples."""
    return [
        sample
        for sample in dataset
        if overwrite or not (predictions_dir / sample.sample_id / ".SUCCESS").is_file()
    ]


def infer(config: DictConfig) -> None:
    """Instantiate 3D dataset & estimator, filter pending samples, and run prediction."""
    log.info(f"Instantiating 3D dataset <{config.data._target_}>")
    dataset: PBREstimationDataset3D = instantiate(config.data)

    log.info(f"Instantiating 3D estimator <{config.method_3d._target_}>")
    estimator: BaseMaterialEstimator3D = instantiate(
        config.method_3d, project_root=PROJECT_ROOT
    )
    if not isinstance(estimator, BaseMaterialEstimator3D):
        raise TypeError(
            f"Expected BaseMaterialEstimator3D, got {type(estimator).__name__}"
        )

    output_dir = Path(config.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))

    overwrite = bool(config.get("runtime", {}).get("overwrite", False))
    debug = bool(config.get("runtime", {}).get("debug", False))

    pending = get_pending_samples(dataset, predictions_dir, overwrite=overwrite)
    log.info(f"{estimator.name}: {len(pending)} pending 3D samples")
    if not pending:
        log.info("All samples are complete (.SUCCESS markers found). Exiting.")
        return

    started = time.time()
    estimator.setup()
    try:
        samples_by_id = {sample.sample_id: sample for sample in pending}
        predictions = estimator.predict_over_dataset(pending, predictions_dir)
        for prediction in predictions:
            if not isinstance(prediction, Prediction3D):
                raise TypeError(
                    f"{estimator.name}.predict_over_dataset() yielded "
                    f"{type(prediction).__name__}, expected Prediction3D"
                )
            if prediction.pbr_asset_glb is None or not Path(prediction.pbr_asset_glb).is_file():
                raise FileNotFoundError(
                    f"{estimator.name} predicted asset missing: {prediction.pbr_asset_glb}"
                )
            sample_dir = predictions_dir / prediction.sample_id
            estimator.align_to_original_uv(
                samples_by_id[prediction.sample_id],
                prediction,
                sample_dir,
                debug=debug,
            )
            (sample_dir / ".SUCCESS").touch()

        for sample in pending:
            sample_dir = predictions_dir / sample.sample_id
            if not (sample_dir / ".SUCCESS").is_file():
                raise FileNotFoundError(
                    f"{estimator.name} returned no prediction for {sample.sample_id}"
                )

        elapsed = time.time() - started
        log.info(
            f"Completed {len(pending)} predictions with {estimator.name} "
            f"in {elapsed:.2f}s ({elapsed / len(pending):.2f}s/sample)"
        )
    finally:
        estimator.teardown()


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer_pbr_3d")
def main(config: DictConfig) -> None:
    infer(config)


if __name__ == "__main__":
    main()
