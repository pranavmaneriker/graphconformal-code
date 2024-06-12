import glob
import logging
import os
import sys
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import psutil
import pyrallis.cfgparsing as pyr_c
import torch
from dgl.dataloading import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info

from .config import BaseExptConfig, ConfExptConfig, SharedBaseConfig
from .conformal_predictor import (
    ConformalMethod,
    ScoreMultiSplitConformalClassifier,
    ScoreSplitConformalClassifer,
)
from .constants import (
    ALL_OUTPUTS_FILE,
    BASEGNN_CKPT_CONFIG_FILE,
    BASEGNN_CKPT_PREFIX,
    CPU_AFF,
    LABELS_KEY,
    NODE_IDS_KEY,
    PROBS_KEY,
    PYTORCH_PRECISION,
    conf_metric_names,
    sample_type,
)
from .custom_logger import CustomLogger
from .data_module import DataModule
from .models import GNN

logging.basicConfig(level=logging.INFO)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def dl_affinity_setup(dl: DataLoader, avl_affinities: Union[List[int], None] = None):
    # setup cpu affinity for dgl dataloader
    # TODO: multi node issues
    if avl_affinities is None:
        avl_affinities = psutil.Process().cpu_affinity()
    assert avl_affinities is not None, "No available cpu affinities"

    cx = getattr(dl, CPU_AFF)
    cx_fn = partial(
        cx,
        loader_cores=avl_affinities[: dl.num_workers],
        compute_cores=avl_affinities[dl.num_workers :],
        verbose=False,
    )
    # cx_fn = partial(cx, verbose=False)
    return cx_fn


def enter_cpu_cxs(
    datamodule: L.LightningDataModule,
    dl_strs: List[str],
    stack: ExitStack,
    num_workers: int,
):
    """Enter cpu contexts on stack and return dataloaders"""
    dls = []
    avl_affinities = psutil.Process().cpu_affinity()
    with suppress_stdout():
        for dl_str in dl_strs:
            dl = getattr(datamodule, dl_str)()
            if num_workers:
                stack.enter_context(dl_affinity_setup(dl, avl_affinities)())
            dls.append(dl)
    return dls


# Helper functions for base gnn
def set_seed_and_precision(seed: int):
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(PYTORCH_PRECISION)


def _get_output_directory(output_directory, dataset, job_id) -> str:
    return os.path.join(output_directory, dataset, job_id)


def prepare_datamodule(args: SharedBaseConfig) -> DataModule:
    rank_zero_info("Setting up data module")
    datamodule = DataModule(
        name=args.dataset,
        seed=args.seed,
        num_workers=min(args.num_workers, args.resource_config.cpus),
        batch_size=args.batch_size,
        dataset_directory=args.dataset_dir,
        use_ddp=args.use_ddp,
    )

    datamodule.prepare_data()
    if args.dataset_loading_style == sample_type.split.name:
        assert (
            args.dataset_split_fractions is not None
        ), f"Dataset split fractions must be provided for loading `{sample_type.split.name}`"

    datamodule.setup(args)

    rank_zero_info("Finished setting up data module")
    return datamodule


def setup_base_model(args: BaseExptConfig, datamodule: DataModule) -> GNN:
    rank_zero_info("Setting up lightning module")
    model = GNN(
        config=args.base_gnn,
        num_features=datamodule.num_features,
        num_classes=datamodule.num_classes,
    )
    rank_zero_info("Finished setting up lightning module")
    return model


def _get_ckpt_dir_fname(output_dir, dataset, job_id, ckpt_prefix) -> Tuple[str, str]:
    ckpt_dir = os.path.join(output_dir, dataset, job_id)
    ckpt_filename = f"{ckpt_prefix}_{{val_acc:.4f}}"
    return ckpt_dir, ckpt_filename


def get_base_ckpt_dir_fname(output_dir, dataset, job_id) -> Tuple[str, str]:
    return _get_ckpt_dir_fname(output_dir, dataset, job_id, BASEGNN_CKPT_PREFIX)


def set_conf_ckpt_dir_fname(
    args: ConfExptConfig, conformal_method_str
) -> Tuple[str, str]:
    if not args.confgnn_config.ckpt_dir or args.confgnn_config.ckpt_filename:
        args.confgnn_config.ckpt_dir, args.confgnn_config.ckpt_filename = (
            _get_ckpt_dir_fname(
                args.output_dir, args.dataset, args.job_id, conformal_method_str
            )
        )
    return args.confgnn_config.ckpt_dir, args.confgnn_config.ckpt_filename


def output_basegnn_config(output_dir: str, args: BaseExptConfig):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, BASEGNN_CKPT_CONFIG_FILE), "w") as f:
        pyr_c.dump(args, f)


def load_basegnn_config_from_ckpt(
    ckpt_dir: str, default_args: Optional[BaseExptConfig] = None
) -> BaseExptConfig:
    """Default args used if yaml not found"""
    yaml_path = os.path.join(ckpt_dir, BASEGNN_CKPT_CONFIG_FILE)
    logging.info(f"Attempting basegnn config load from {yaml_path}")
    if os.path.exists(yaml_path):
        if default_args is not None:
            logging.warning(
                f"Config will be overwritten by existing {default_args.dataset}/{default_args.job_id}"
            )
        # load BaseExpt config from yaml
        with open(yaml_path, "r") as f:
            return pyr_c.load(BaseExptConfig, f)
    else:
        assert (
            default_args is not None
        ), "No default args provided and no config file found"
        return default_args


def _base_ckpt_path(job_output_dir: str):
    return glob.glob(os.path.join(job_output_dir, f"{BASEGNN_CKPT_PREFIX}*.ckpt"))


def set_trained_basegnn_path(args: ConfExptConfig, ckpt_dir: Optional[str] = None):
    if ckpt_dir is not None:
        args.confgnn_config.base_model_path = _base_ckpt_path(ckpt_dir)[0]
    elif not args.confgnn_config.base_model_path:
        job_output_dir = _get_output_directory(
            args.output_dir, args.dataset, args.base_job_id
        )
        # TODO: We assume that the first checkpoint found in the job id will be the one to use
        args.confgnn_config.base_model_path = _base_ckpt_path(job_output_dir)[0]
    return args.confgnn_config.base_model_path


def load_basegnn(ckpt_dir: str, args: BaseExptConfig, datamodule) -> GNN:
    base_ckpt_path = _base_ckpt_path(ckpt_dir)
    if args.resume_from_checkpoint:
        if len(base_ckpt_path) > 0:
            base_ckpt_path = base_ckpt_path[0]
            logging.info(f"Resuming from checkpoint: {base_ckpt_path}")
            model = GNN.load_from_checkpoint(base_ckpt_path)
            return model
        else:
            logging.warning("No checkpoint found for resuming. Training from scratch.")
    return setup_base_model(args, datamodule)


def setup_trainer(
    args: SharedBaseConfig,
    expt_logger: Optional[CustomLogger] = None,
    /,
    strategy="auto",
    callbacks=None,
    plugins=None,
    **kwargs,
) -> L.Trainer:
    rank_zero_info("Setting up trainer")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.resource_config.gpus,
        num_nodes=args.resource_config.nodes,
        max_epochs=args.epochs,
        strategy=strategy,
        callbacks=callbacks,
        plugins=plugins,
        logger=expt_logger,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        **kwargs,
    )
    rank_zero_info("Finished setting up trainer")
    return trainer


def output_basegnn_results(args: BaseExptConfig, results: Dict[str, torch.Tensor]):
    assert NODE_IDS_KEY in results
    # assert that results[NODE_IDS_KEY] is sorted
    assert torch.all(
        results[NODE_IDS_KEY].argsort() == torch.arange(len(results[NODE_IDS_KEY]))
    )
    assert LABELS_KEY in results and PROBS_KEY in results

    job_output_dir = _get_output_directory(args.output_dir, args.dataset, args.job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    torch.save(results, os.path.join(job_output_dir, ALL_OUTPUTS_FILE))


def run_basegnn_inference_alldl(
    model: GNN, trainer: L.Trainer, ckpt_path: str, datamodule: DataModule
):
    with ExitStack() as stack:
        dl = enter_cpu_cxs(
            datamodule, ["all_dataloader"], stack, datamodule.num_workers
        )
        trainer.test(model, dataloaders=dl, ckpt_path=ckpt_path, verbose=False)
        return model.latest_test_results


def check_sampling_consistent(
    base_expt_config: BaseExptConfig, expt_config: ConfExptConfig
):
    assert base_expt_config.dataset == expt_config.dataset, "Dataset must be consistent"
    assert base_expt_config.seed == expt_config.seed, "Seed must be consistent"
    assert (
        base_expt_config.dataset_loading_style == expt_config.dataset_loading_style
    ), "Dataset loading style must be consistent"
    if base_expt_config.dataset_loading_style == sample_type.split.name:
        assert (
            base_expt_config.dataset_split_fractions.train
            == expt_config.dataset_split_fractions.train
            and base_expt_config.dataset_split_fractions.valid
            == expt_config.dataset_split_fractions.valid
        ), "Dataset train and validation split fractions must be consistent"
    elif base_expt_config.dataset_loading_style == sample_type.n_samples_per_class.name:
        assert (
            base_expt_config.dataset_n_samples_per_class
            == expt_config.dataset_n_samples_per_class
        ), "Number of samples per class must be consistent"


def load_basegnn_outputs(args: ConfExptConfig, job_output_dir: Optional[str] = None):
    if not job_output_dir:
        job_output_dir = _get_output_directory(
            args.output_dir, args.dataset, args.base_job_id
        )
    results = torch.load(os.path.join(job_output_dir, ALL_OUTPUTS_FILE))
    probs, labels = results[PROBS_KEY], results[LABELS_KEY]
    assert isinstance(probs, torch.Tensor) and isinstance(labels, torch.Tensor)
    return probs, labels


def update_dataclass_from_dict(dataclass, update_dict):
    for k, v in update_dict.items():
        if hasattr(dataclass, k):
            setattr(dataclass, k, v)
        else:
            logging.warning(
                f"Attempted to update {k} in {type(dataclass)} but it does not exist."
            )


def run_conformal(
    args: ConfExptConfig,
    datamodule: DataModule,
    expt_logger: CustomLogger,
    base_ckpt_dir: Optional[str] = None,
):
    # Load probs/labels from base
    probs, labels = load_basegnn_outputs(args, base_ckpt_dir)
    assert (
        probs.shape[1] == datamodule.num_classes
    ), f"Loaded probs has {probs.shape[1]} classes, but the dataset has {datamodule.num_classes} classes"

    # note that splits are setup but not sampler
    conformal_method = ConformalMethod(args.conformal_method)
    match conformal_method:
        # TODO: move individual cases into separate functions in utils
        case ConformalMethod.TPS | ConformalMethod.APS | ConformalMethod.NAPS:
            cp = ScoreSplitConformalClassifer(config=args, datamodule=datamodule)

            split_conf_input = args.primitive_config
            if conformal_method == ConformalMethod.NAPS:
                split_conf_input = args.neighborhood_config

            pred_sets, test_labels = cp.run(
                probs=probs,
                labels=labels,
                split_conf_input=split_conf_input,
            )

        case ConformalMethod.DAPS | ConformalMethod.DTPS | ConformalMethod.RAPS:
            cp = ScoreMultiSplitConformalClassifier(config=args, datamodule=datamodule)

            split_conf = (
                args.raps_config
                if conformal_method == ConformalMethod.RAPS
                else args.diffusion_config
            )

            pred_sets, test_labels = cp.run(
                probs=probs, labels=labels, split_conf_input=split_conf
            )

            if cp.best_params is not None:
                expt_logger.log_hyperparams(cp.best_params)

        case ConformalMethod.CFGNN:
            assert (
                args.confgnn_config is not None
            ), f"confgnn_config cannot be None for CFGNN"
            _ = set_trained_basegnn_path(args, base_ckpt_dir)
            _, _ = set_conf_ckpt_dir_fname(args, conformal_method.value)
            cp = ScoreMultiSplitConformalClassifier(config=args, datamodule=datamodule)

            pred_sets, test_labels = cp.run(
                split_conf_input=args.confgnn_config, logger=expt_logger, probs=probs
            )

        case _:
            raise NotImplementedError
    return pred_sets, test_labels
