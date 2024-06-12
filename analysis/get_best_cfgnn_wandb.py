import logging
import os

import pandas as pd
import wandb
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "../configs/custom_configs/best_cfgnn_configs"
DATASETS = [
    "CiteSeer",
    "PubMed",
    "Cora",
    "Coauthor_CS",
    "Coauthor_Physics",
    "Amazon_Photos",
    "Amazon_Computers",
    "Flickr",
    "ogbn-arxiv",
    "ogbn-products",
]
SPLIT_STYLES = ["split", "n_samples_per_class"]
SPLIT_FRACS = [(0.2, 0.1), (0.2, 0.2), (0.3, 0.1), (0.3, 0.2)]  # train, val
N_SAMPLES = [10, 20, 40, 80]


def runs_to_result_df(runs):
    all_results = []
    for run in tqdm(runs):
        hist = run.history()
        conf = run.config
        run_name = conf["logging_config"]["wandb_config"]["run_name"]
        job_id = conf["job_id"]
        sample_id = run_name[-6:]
        run_prefix = run_name[:-6]
        if "confgnn_val_eff" in hist.columns:
            all_results.append(
                {
                    "full_config": conf,
                    "confgnn_val_eff": hist["confgnn_val_eff"].max(),
                    "job_id": job_id,
                    "run_name": run_name,
                    "sample_id": sample_id,
                    "run_prefix": run_prefix,
                }
            )
    return pd.DataFrame(all_results)


def get_split_results_df(
    dataset, train_split, valid_split, runs_path="avoirpp/avoir++"
):
    api = wandb.Api()
    runs_filter = {
        "config.logging_config.wandb_config.job_type": "tune",
        "config.dataset": dataset,
        "config.dataset_loading_style": "split",
        "config.dataset_split_fractions.train": train_split,
        "config.dataset_split_fractions.valid": valid_split,
        # "config.logging_config.wandb_config.run_name": {"$regex": "[a-z]*-[a-z]*-[0-9]*"}
    }
    runs = api.runs(runs_path, runs_filter)
    # table isnt always stored because jobs failed so find best confrig
    return runs_to_result_df(runs)


def get_nsamples_results_df(dataset, n_samples_per_class, runs_path="avoirpp/avoir++"):
    api = wandb.Api()
    runs_filter = {
        "config.logging_config.wandb_config.job_type": "tune",
        "config.dataset": dataset,
        "config.dataset_loading_style": "n_samples_per_class",
        "config.dataset_n_samples_per_class": n_samples_per_class,
        # "config.logging_config.wandb_config.run_name": {"$regex": "[a-z]*-[a-z]*-[0-9]*"}
    }
    runs = api.runs(runs_path, runs_filter)
    # table isnt always stored because jobs failed so find best confrig
    return runs_to_result_df(runs)


def output_best_config_for_split(dataset, train_split, valid_split, config_output_dir):
    logging.info(f"Getting best config for {dataset} {train_split} {valid_split}")
    df = get_split_results_df(dataset, train_split, valid_split)
    if len(df) == 0:
        logging.warning(f"No results found for {dataset} {train_split} {valid_split}")
        return
    best_jid, best_run = (
        df.groupby(["job_id", "run_name"])["confgnn_val_eff"].mean().idxmin()
    )
    best_config = df[(df["job_id"] == best_jid) & (df["run_name"] == best_run)][
        "full_config"
    ].values[0]

    split_dir = os.path.join(
        config_output_dir, dataset, "split", f"{train_split}_{valid_split}"
    )
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, "cfgnn_config.yaml")
    # reset the seed to 0
    best_config["seed"] = 0
    with open(split_path, "w") as f:
        yaml.dump(best_config, f, indent=2)
    logging.info(
        f"Output best config for {dataset} {train_split} {valid_split} to {split_path}"
    )


def output_best_config_for_nsamples(dataset, n_samples_per_class, config_output_dir):
    logging.info(f"Getting best config for {dataset} {n_samples_per_class}")
    df = get_nsamples_results_df(dataset, n_samples_per_class)
    if len(df) == 0:
        logging.warning(f"No results found for {dataset} {n_samples_per_class}")
        return
    best_jid, best_run = (
        df.groupby(["job_id", "run_name"])["confgnn_val_eff"].mean().idxmin()
    )
    best_config = df[(df["job_id"] == best_jid) & (df["run_name"] == best_run)][
        "full_config"
    ].values[0]

    nsamples_dir = os.path.join(
        config_output_dir, dataset, "n_samples_per_class", f"{n_samples_per_class}"
    )
    os.makedirs(nsamples_dir, exist_ok=True)
    nsamples_path = os.path.join(nsamples_dir, "cfgnn_config.yaml")
    # reset the seed to 0
    best_config["seed"] = 0
    with open(nsamples_path, "w") as f:
        yaml.dump(best_config, f, indent=2)
    logging.info(
        f"Output best config for {dataset} {n_samples_per_class} to {nsamples_path}"
    )


if __name__ == "__main__":
    for dataset in DATASETS:
        for split_style in SPLIT_STYLES:
            if split_style == "split":
                for train_split, valid_split in SPLIT_FRACS:
                    output_best_config_for_split(
                        dataset, train_split, valid_split, OUTPUT_DIR
                    )
            else:
                for n_samples in N_SAMPLES:
                    output_best_config_for_nsamples(dataset, n_samples, OUTPUT_DIR)
