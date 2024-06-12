import logging
import os

import graph_conformal.utils as utils
import pandas as pd
import pyrallis.argparsing as pyr_a
import torch
from graph_conformal.conf_metrics import compute_metric
from graph_conformal.config import ConfExptConfig
from graph_conformal.constants import conf_metric_names
from graph_conformal.custom_logger import CustomLogger

# not adding this to constants since this is only a utility for easy wandb plots
CUSTOM_STEP = "custom_step"  # helper for wandb plots
tern_logger = logging.getLogger(__name__)


def main() -> None:
    args = pyr_a.parse(config_class=ConfExptConfig)

    # make sure that the sampling config for the expt is same as that of the base job
    base_ckpt_dir, _ = utils.get_base_ckpt_dir_fname(
        args.output_dir, args.dataset, args.base_job_id
    )
    base_expt_config = utils.load_basegnn_config_from_ckpt(base_ckpt_dir)
    utils.check_sampling_consistent(base_expt_config, args)

    # setup dataloaders
    utils.set_seed_and_precision(args.seed)
    datamodule = utils.prepare_datamodule(args)
    # reshulle the calibration and test sets if required
    datamodule.resplit_calib_test(args)

    expt_logger = CustomLogger(args.logging_config)
    expt_logger.log_hyperparams(vars(args))

    pred_sets, test_labels = utils.run_conformal(args, datamodule, expt_logger)

    metrics_to_compute = args.conformal_metrics
    feature_idx = args.conformal_feature_idx
    test_features = None
    if feature_idx is not None:
        test_features = datamodule.get_test_nodes_features(feature_idx)

    all_metrics = {}
    for metric in metrics_to_compute:
        metric_val = compute_metric(
            metric, pred_sets, test_labels, args.alpha, test_features
        )
        if isinstance(metric_val, torch.Tensor) and len(metric_val.size()) == 0:
            metric_val = metric_val.item()
        match metric:
            # case conf_metrics.set_sizes.name: expt_logger.log_histogram(f"{conformal_method.value}_{metric}_histogram", f"{conformal_method.value}_{metric}", metric_val)
            case conf_metric_names.set_sizes.name:
                expt_logger.log_histogram(
                    f"{metric}_histogram",
                    f"{metric}",
                    metric_val,
                )
            case _ if isinstance(metric_val, float):
                all_metrics[metric] = [metric_val]
                expt_logger.log_metrics({f"{metric}": metric_val, CUSTOM_STEP: 0})
            case _:
                print(
                    f"Unsupported value {metric_val} for `{metric}` when using `{args.conformal_method}`"
                )

    all_metrics = pd.DataFrame(all_metrics)
    # params = pd.DataFrame(expt_logger.params)
    out_dir = os.path.join(args.results_output_dir, f"alpha_{args.alpha}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"cfgnn.csv")
    # out_params = os.path.join(out_dir, f"cfgnn_params.csv")
    all_metrics.to_csv(out_file, index=False)
    # params.to_csv(out_params, index=False)


if __name__ == "__main__":
    # python run_conformal.py --config_path="configs/tps_default.yaml"
    main()
