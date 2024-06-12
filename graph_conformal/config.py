import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .constants import (
    CLASSIFICATION_DATASETS,
    CORA,
    ConformalMethod,
    conf_metric_names,
    layer_types,
    sample_type,
)


# helper for logging dataclasses as dict
# see https://stackoverflow.com/a/51286749
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class BaseGNNConfig:
    """Config for the base GNN model and its training."""

    # Learning rate
    lr: float = field(default=0.01)
    # Model layer type
    model: str = field(default=layer_types.GCN.name)
    # Number of hidden channels
    hidden_channels: int = field(default=16)
    # Number of heads for GAT
    heads: int = field(default=1)
    # Number of layers
    layers: int = field(default=2)
    # Aggregation method
    aggr: str = field(default="mean")
    # Dropout prob
    dropout: float = field(default=0.5)
    # Fanout for neighbor sampling
    fanouts: List[int] = field(default_factory=list)

    def __post_init__(self):
        ltypes = [lt.name for lt in layer_types]
        assert (
            self.model in ltypes
        ), f"Invalid model type {self.model}, must be in {ltypes}."
        # assert self.aggr in ['mean', 'add', 'max'], f"Invalid aggregation method {self.aggr}."
        # TODO: Optimal parammeter loading


@dataclass
class WandBConfig:
    project_name: str = field(default="avoir++")
    log_model: bool = field(default=False)
    # run name
    run_name: Optional[str] = field(default=None)
    # Group base vs conformal expts ("base" or "conformal")
    group: str = field(default="base")
    # job type tag added to runs. "debug" for debugging runs, "expt" for main runs, tune for tuning runs
    job_type: str = field(default="debug")
    # Flag to make W&B logging offline. Offline runs can be synced later with `wandb sync`
    offline: bool = field(default=False)


@dataclass
class LoggingConfig:
    """Config for custom logger"""

    # whether to use wandb
    use_wandb: bool = field(default=False)
    use_tensorboard: bool = field(default=False)
    use_terminal: bool = field(default=False)
    # Config for wandb
    wandb_config: Optional[WandBConfig] = field(default=None)


@dataclass
class DatasetSplitConfig:
    train: float = field(default=0.2)
    valid: float = field(default=0.1)
    calib: float = field(default=0.35)


@dataclass
class ResourceConfig:
    cpus: int = field(default=1)
    gpus: int = field(default=0)
    nodes: int = field(default=1)
    custom: Dict[str, int] = field(default_factory=dict)


@dataclass
class SharedBaseConfig:
    """Overall config for the experiment."""

    # Random seed
    seed: int = field(default=0)
    # dataset name
    dataset: str = field(default=CORA)
    # dataset loading style
    dataset_loading_style: str = field(default=sample_type.split.name)
    # split fractions (train/valid/calib)
    dataset_split_fractions: Optional[DatasetSplitConfig] = field(
        default_factory=DatasetSplitConfig
    )
    # samples per class
    dataset_n_samples_per_class: Optional[int] = field(default=None)
    # output directory for results
    output_dir: str = field(default="./outputs")
    # dataset directory for dgl datasets
    dataset_dir: str = field(default="./datasets")
    # number of workers for dataloader
    num_workers: int = field(default=0)
    # SLURM job id or current date if not provided
    job_id: str = field(default=datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))
    # Logging config
    logging_config: Optional[LoggingConfig] = field(default=None)
    # Resource config
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)
    # Batch size for training
    batch_size: int = field(default=256)
    # Number of epochs for training
    epochs: int = field(default=100)
    # Wheter to use Distributed Data Parallelism (DDP)
    use_ddp: Optional[bool] = field(default=None)

    def __post_init__(self):
        if self.use_ddp is None:
            self.use_ddp = (
                self.resource_config.nodes > 1 or self.resource_config.gpus > 1
            )


@dataclass
class BaseExptConfig(SharedBaseConfig):
    """Overall config for the base model training."""

    # model config
    base_gnn: BaseGNNConfig = field(default_factory=BaseGNNConfig)

    # whether to resume from checkpoint (searches OUTPUT_DIRECTORY/dataset/job_id)
    resume_from_checkpoint: Optional[bool] = field(default=True)

    def __post_init__(self):
        sample_types = [st.name for st in sample_type]
        assert (
            self.dataset_loading_style in sample_types
        ), f"Invalid dataset loading style {self.dataset_loading_style}, must be in {sample_types}."
        assert (
            self.dataset in CLASSIFICATION_DATASETS
        ), f"Invalid dataset {self.dataset}, must be in {CLASSIFICATION_DATASETS}."


@dataclass
class PrimitiveScoreConfig:
    """
    Configs for primitive score functions like APS and TPS
    """

    use_aps_epsilon: Optional[bool] = field(default=True)
    use_tps_classwise: Optional[bool] = field(default=False)


@dataclass
class MultiSplitTuneFractionConfig:
    # fraction of dataset used for tuning hyperparams
    tuning_fraction: float = field(default=0.5)


@dataclass
class ConfGNNConfig(BaseGNNConfig, PrimitiveScoreConfig, MultiSplitTuneFractionConfig):
    """Config for the conformal GNN model."""

    # path to the base mode
    base_model_path: str = field(default="")
    # directory for checkpointing
    ckpt_dir: str = field(default="")
    # filename for checkpointing
    ckpt_filename: str = field(default="")

    # load saved probs instead of stacking the models
    load_probs: bool = field(default=False)

    # use tps or aps for training
    train_fn: str = field(default="tps")
    # tps/aps for eval
    eval_fn: str = field(default="aps")
    # use aps epsilon when one of the functions is aps
    use_aps_epsilon: bool = field(default=True)
    # fraction of epochs to use only CrossEntropy for training
    label_train_fraction: float = field(default=0.5)
    # weight for CrossEntropy loss (conformal wt is 1 - ce_weight)
    ce_weight: float = field(default=0.5)
    # temperature
    temperature: float = field(default=0.5)
    # batch size during eval for faster eval
    # test_batch_size: int = field(default=-1)

    # use origianl rule of N_cal = min{1000, #test / 2}
    limit_calibration_set: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.train_fn in ["tps", "aps"], f"Invalid train_fn {self.train_fn}."
        assert self.eval_fn in ["tps", "aps"], f"Invalid eval_fn {self.eval_fn}."


@dataclass
class CVConformalHyperparamConfig(PrimitiveScoreConfig, MultiSplitTuneFractionConfig):
    # number of iterations to perform when tuning diffusion_param
    n_iterations: int = field(default=20)

    # whether each iteration should use a different portion of calib set for tuning
    resplit_every_iteration: bool = field(default=False)


@dataclass
class RegularizedConfig(CVConformalHyperparamConfig):
    """
    Config for RAPS method
    """

    # whether to use original raps random adjustment or fix it
    # the original raps uses - u * \pi_L during calibratio only
    raps_mod: bool = field(default=False)


@dataclass
class DiffusionConfig(CVConformalHyperparamConfig):
    """
    Configs for diffusion transformation and diffusion parameter tuning
    """

    pass


@dataclass
class NeighborhoodConfig(PrimitiveScoreConfig):
    """
    Configs for neighborhood transformation
    """

    k_hop_neighborhood: Optional[int] = field(default=2)

    weight_function: Optional[str] = field(default="uniform")

    num_batches: Optional[int] = field(default=5)


@dataclass
class ConfExptConfig(SharedBaseConfig):
    # conformal seed - use a different split of the calib/test set than the base model
    # useful for multiple runs to study coverage distribution plots
    conformal_seed: Optional[int] = field(default=None)
    # whether to use the outputs of a base job_id (True)
    # or run the full data through the base gnn (False)
    # load_probs_from_outputs: bool = field(default=True)
    # base job_id (for loading probs)
    base_job_id: Optional[str] = field(default=None)
    # desired alpha level
    alpha: float = field(default=0.1)
    # conformal method to run
    conformal_method: str = field(default="tps")
    # List of conformal metrics to compute
    conformal_metrics: List[str] = field(
        default_factory=lambda: [cm.name for cm in conf_metric_names]
    )
    # feature for feature stratified coverage
    conformal_feature_idx: Optional[int] = field(default=None)

    """Conformal method specific arguments"""
    # neighborhood_config
    neighborhood_config: Optional[NeighborhoodConfig] = field(
        default_factory=NeighborhoodConfig
    )
    # primitive_config
    primitive_config: Optional[PrimitiveScoreConfig] = field(
        default_factory=PrimitiveScoreConfig
    )
    # confgnn config
    confgnn_config: Optional[ConfGNNConfig] = field(default_factory=ConfGNNConfig)
    # diffusion transformation config
    diffusion_config: Optional[DiffusionConfig] = field(default_factory=DiffusionConfig)
    # RAPS config
    raps_config: Optional[RegularizedConfig] = field(default_factory=RegularizedConfig)
    # output directory for conformal results
    results_output_dir: str = field(default="./conformal_results")
    # whether to resume from checkpoint (searches OUTPUT_DIRECTORY/dataset/job_id)
    resume_from_checkpoint: Optional[bool] = field(default=True)

    def __post__init__(self):
        conf_metric_names = [cm.name for cm in conf_metric_names]
        assert self.conformal_metrics is None or all(
            [cm in conf_metric_names for cm in self.conformal_metrics]
        ), f"Invalid conformal metrics {self.conformal_metrics}."
        assert (
            self.base_job_id is not None if self.load_probs_from_outputs else True
        ), "Need to provide base_job_id if load_probs_from_jobid is True."
        conformal_methods = [st.name for st in ConformalMethod]
        assert (
            self.conformal_method in conformal_methods
        ), f"Invalid conformal method {self.conformal_method}, must be in {conformal_methods}."

        if self.conformal_method == ConformalMethod.APS:
            assert (
                self.primitive_config is not None
                and self.primitive_config.use_aps_epsilon is not None
            ), "Need to provide use_aps_epsilon for APS methods."

        if self.conformal_method == ConformalMethod.RAPS:
            assert self.raps_config is not None, "Need to provide raps_config for RAPS."

        if self.conformal_method == ConformalMethod.CFGNN:
            assert (
                self.confgnn_config is not None
            ), "Need to provide confgnn_config for CFGNN."

        if self.conformal_method in [ConformalMethod.DAPS, ConformalMethod.DTPS]:
            assert (
                self.diffusion_config is not None
            ), "Need to diffusion_config for diffusion-based methods"
            if self.conformal_method == ConformalMethod.DAPS:
                assert (
                    self.diffusion_config.use_aps_epsilon is not None
                ), "Need to use_aps_epsilon for DAPS"

        if self.conformal_method in [ConformalMethod.NAPS]:
            assert (
                self.neighborhood_config is not None
            ), "Need neighborhood_config for neighborhood-based methods"
            if self.conformal_method == ConformalMethod.NAPS:
                assert (
                    self.neighborhood_config.use_aps_epsilon is not None
                ), "Need to use_aps_epsilon for NAPS"


SplitConfInput = (
    PrimitiveScoreConfig
    | DiffusionConfig
    | RegularizedConfig
    | NeighborhoodConfig
    | ConfGNNConfig
)
