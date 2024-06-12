import logging
import os
import traceback

import graph_conformal.utils as utils
import pandas as pd
import pyrallis.argparsing as pyr_a
import pyrallis.cfgparsing as pyr_c
import torch
from graph_conformal.conf_metrics import compute_metric
from graph_conformal.config import ConfExptConfig, LoggingConfig
from graph_conformal.constants import sample_type
from graph_conformal.custom_logger import CustomLogger
from hpt_config import ConfTrialsExptConfig
from tqdm import tqdm

## not adding this to constants since this is only a utility for easy wandb plots
CUSTOM_STEP = "custom_step"  # helper for wandb plots
term_logger = logging.getLogger(__name__)


class DictLogger(CustomLogger):
    def __init__(self, config: LoggingConfig | None) -> None:
        super().__init__(config)
        self.logs = []
        self.params = []

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        self.logs.append({"step": step, "metrics": metrics})
        super().log_metrics(metrics, step)

    def log_hyperparams(self, hparams: dict) -> None:
        self.params.append(hparams)
        super().log_hyperparams(hparams)


def run_trials(
    args: ConfTrialsExptConfig, expt_config: ConfExptConfig, config_prefix: str
):
    base_ckpt_dir = args.base_model_dir
    n_trials = args.trials_per_config
    # make sure that the sampling config for the expt is same as that of the base job
    base_expt_config = utils.load_basegnn_config_from_ckpt(base_ckpt_dir)
    utils.set_seed_and_precision(base_expt_config.seed)
    # fix params for conf model
    expt_config.dataset = base_expt_config.dataset
    expt_config.dataset_loading_style = base_expt_config.dataset_loading_style
    expt_config.dataset_split_fractions = base_expt_config.dataset_split_fractions
    expt_config.dataset_n_samples_per_class = (
        base_expt_config.dataset_n_samples_per_class
    )
    expt_config.seed = base_expt_config.seed
    expt_config.base_job_id = base_expt_config.job_id
    # TODO: dataset dir may need to be fixed
    datamodule = utils.prepare_datamodule(expt_config)
    logging.info(
        f"Running {n_trials} trials for {expt_config.dataset} with "
        f"{config_prefix} method and {expt_config.alpha} alpha."
    )

    all_metrics = []
    for trial in tqdm(range(n_trials)):
        trial_metrics = {}
        seed = trial
        expt_config.conformal_seed = seed
        # setup dataloaders
        # reshulle the calibration and test sets if required
        datamodule.resplit_calib_test(expt_config)

        expt_logger = DictLogger(expt_config.logging_config)
        expt_logger.log_hyperparams(vars(args))

        pred_sets, test_labels = utils.run_conformal(
            expt_config, datamodule, expt_logger, base_ckpt_dir
        )

        metrics_to_compute = expt_config.conformal_metrics
        feature_idx = expt_config.conformal_feature_idx
        test_features = None
        if feature_idx is not None:
            test_features = datamodule.get_test_nodes_features(feature_idx)

        for metric in metrics_to_compute:
            metric_val = compute_metric(
                metric,
                pred_sets,
                test_labels,
                features=test_features,
                alpha=expt_config.alpha,
            )
            if isinstance(metric_val, torch.Tensor) and len(metric_val.size()) == 0:
                metric_val = metric_val.item()
            match metric:
                # case conf_metrics.set_sizes.name: expt_logger.log_histogram(f"{conformal_method.value}_{metric}_histogram", f"{conformal_method.value}_{metric}", metric_val)
                # case conf_metric_names.set_sizes.name:
                #    expt_logger.log_histogram(
                #        f"{metric}_histogram",
                #        f"{metric}",
                #        metric_val,
                #    )
                case _ if isinstance(metric_val, float):
                    trial_metrics[metric] = metric_val
                    expt_logger.log_metrics({f"{metric}": metric_val, CUSTOM_STEP: 0})
                case _:
                    pass
                    # print(
                    #    f"Unsupported value {metric_val} for `{metric}` when using `{args.conformal_method}`"
                    # )
        all_metrics.append(trial_metrics)
    return all_metrics, expt_logger.params


def main() -> None:
    args: ConfTrialsExptConfig = pyr_a.parse(config_class=ConfTrialsExptConfig)

    expt_configs_dir = args.expt_configs_dir
    out_dir = os.path.join(args.results_output_dir, f"alpha_{args.alpha}")
    os.makedirs(out_dir, exist_ok=True)

    for config_file in os.listdir(expt_configs_dir):
        try:
            config_prefix = config_file.split(".")[0]
            config_path = os.path.join(expt_configs_dir, config_file)
            with open(config_path, "r") as f:
                expt_config: ConfExptConfig = pyr_c.load(ConfExptConfig, f)
            expt_config.alpha = args.alpha
            if (
                args.calib_test_equal
                and expt_config.dataset_loading_style == sample_type.split.name
            ):
                expt_config.dataset_split_fractions.calib = (
                    (
                        1
                        - expt_config.dataset_split_fractions.train
                        - expt_config.dataset_split_fractions.valid
                    )
                ) / 2
            # each row is metrics for one trial
            metrics, best_params = run_trials(args, expt_config, config_prefix)
            metrics = pd.DataFrame(metrics)
            best_params = pd.DataFrame(best_params)
            out_file = os.path.join(out_dir, f"{config_prefix}.csv")
            out_params = os.path.join(out_dir, f"{config_prefix}_params.csv")
            metrics.to_csv(out_file, index=False)
            best_params.to_csv(out_params, index=False)
        except Exception as e:
            term_logger.exception(f"Error in running {config_file}.")
            # print exception stacktrace
            traceback.print_exc()


if __name__ == "__main__":
    main()
