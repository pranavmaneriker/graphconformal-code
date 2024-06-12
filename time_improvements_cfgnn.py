"""We aim to run two kinds of benchmarking eval
1. Time to specific efficiency (across full batch and different kinds of batch gradient descent, how much time (an how many epochs) does it take for the coonfgnn to reach a specific efficiency.
2. Caching probabilities vs 2 GNN model: Given the batched version, how much faster is it to run the cached version of the probabilities vs running the GNN model by sample n_layers_1 + n_layers_2 layers.
"""
import time
import os
import glob
from dataclasses import dataclass, field
import logging

import torch
import lightning.pytorch as L
from dgl.dataloading import MultiLayerFullNeighborSampler
import pyrallis.argparsing as pyr_a
import pandas as pd

from graph_conformal.constants import conf_metric_names, ConformalMethod, Stage, SCORES_KEY, LABELS_KEY
import graph_conformal.utils as utils
from graph_conformal.custom_logger import CustomLogger
from graph_conformal.config import ConfExptConfig, ConfGNNConfig, LoggingConfig, ResourceConfig
#from graph_conformal.conformal_predictor import ScoreMultiSplitConformalClassifier
from graph_conformal.models import CFGNN
from graph_conformal.transformations import PredSetTransformation
from graph_conformal.conf_metrics import compute_metric


logger= logging.getLogger(__name__)

@dataclass
class RuntimeBenchmarkingConfig:
    n_runs_per_expt: int = 5
    benchmark_output_file: str = "PubMed_runtime_benchmarking.csv"
    base_output_dir: str = field(default="./outputs")
    dataset: str = field(default="PubMed")
    base_job_id: str = field(default="basegnn_pubmed_debug")
    dataset_dir: str = field(default="./datasets")
    conformal_seed: int = field(default=0)
    alpha: float = field(default=0.1)
    
    # the rest of the args are method specific

    max_epochs: int = field(default=1000)
    batch_size: int = field(default=-1)
    train_fn: str = field(default="tps")
    eval_fn: str = field(default="aps")
    load_probs: bool = field(default=False)
    lr: float = field(default=0.001)

def run_and_time_conformal(runtime_benchmark_config: RuntimeBenchmarkingConfig):
                           #base_output_dir, dataset, base_job_id, conformal_seed, alpha, max_epochs, batch_size, dataset_dir):
    """
        Run the CFGNN training loop and measure the time/effeiciency with the given config
    """
    # local vars for shorter code
    base_output_dir = runtime_benchmark_config.base_output_dir
    dataset = runtime_benchmark_config.dataset
    base_job_id = runtime_benchmark_config.base_job_id
    conformal_seed = runtime_benchmark_config.conformal_seed
    alpha = runtime_benchmark_config.alpha
    max_epochs = runtime_benchmark_config.max_epochs
    batch_size = runtime_benchmark_config.batch_size
    dataset_dir = runtime_benchmark_config.dataset_dir
    train_fn = runtime_benchmark_config.train_fn
    eval_fn = runtime_benchmark_config.eval_fn
    load_probs = runtime_benchmark_config.load_probs
    lr = runtime_benchmark_config.lr

    base_ckpt_dir, _ = utils.get_base_ckpt_dir_fname(base_output_dir, dataset, base_job_id)
    base_expt_config = utils.load_basegnn_config_from_ckpt(base_ckpt_dir)
    base_model_path = glob.glob(os.path.join(base_ckpt_dir, "basegnn*.ckpt"))[0]

    assert base_expt_config.dataset_loading_style == "split"
    assert base_expt_config.dataset == dataset

    # fixed resource config for consistent comparison
    resource_config = ResourceConfig(cpus=10, gpus=1, nodes=1)
    n_dl_workers = 1 

    # config for orig
    conf_gnn_config = ConfGNNConfig(model="GCN",
                                    dropout=0.25,
                                    heads=1,
                                    hidden_channels=128,
                                    layers=2,
                                    train_fn=train_fn, # "tps" for base model
                                    eval_fn=eval_fn, # `aps` for all
                                    use_aps_epsilon=False,
                                    label_train_fraction=0.5,
                                    ce_weight=0.5,
                                    temperature=0.5,
                                    load_probs=load_probs,
                                    base_model_path=base_model_path,
                                    ckpt_dir=None)

    # cfgnn_expt setup, some params get copied from base model
    expt_config = ConfExptConfig(
        seed=base_expt_config.seed,
        dataset_loading_style=base_expt_config.dataset_loading_style,
        dataset_split_fractions=base_expt_config.dataset_split_fractions,
        dataset=dataset,
        base_job_id=base_expt_config.job_id,
        confgnn_config=conf_gnn_config,
        conformal_seed=conformal_seed,
        alpha=alpha,
        dataset_dir=dataset_dir,
        resource_config=resource_config,
        batch_size=batch_size,
        num_workers=n_dl_workers)

    # train base model
    conf_gnn_config.lr = lr
    expt_config.epochs = max_epochs

    # setup data
    ## setup random seeds
    utils.set_seed_and_precision(expt_config.seed)
    ## setup dataset/splits
    datamodule = utils.prepare_datamodule(expt_config)
    ## resplit calib/test
    datamodule.resplit_calib_test(expt_config)

    ## dummy logger
    #expt_logger = CustomLogger(config=LoggingConfig())

    # manually run CFGNN
    probs, labels = utils.load_basegnn_outputs(expt_config, base_ckpt_dir)


    # manually carry out the conformal training
    datamodule.split_calib_tune_qscore(tune_frac=conf_gnn_config.tuning_fraction)
    if conf_gnn_config.load_probs:
        datamodule.update_features(probs)
    cfgnn = CFGNN(config=conf_gnn_config, alpha=expt_config.alpha, 
                num_epochs=expt_config.epochs, num_classes=datamodule.num_classes)

    if not conf_gnn_config.load_probs:
        total_layers = conf_gnn_config.layers + cfgnn.base_model_num_layers
    else:
        total_layers = conf_gnn_config.layers

    # fast traininer, no validation
    trainer = L.Trainer(
        accelerator="gpu",
        devices=expt_config.resource_config.gpus,
        num_nodes=expt_config.resource_config.nodes,
        max_epochs=expt_config.epochs,
        enable_checkpointing=False,
        logger=False
    )

    # run epochs
    split_dict = datamodule.split_dict
    calib_tune_nodes = split_dict[Stage.CALIBRATION_TUNE]
    calib_qscore_nodes = split_dict[Stage.CALIBRATION_QSCORE]
    test_nodes = split_dict[Stage.TEST]

    sampler = MultiLayerFullNeighborSampler(total_layers)

    ctd_bs = batch_size
    if batch_size == -1:
        ctd_bs = len(calib_tune_nodes)
    calib_tune_dl = datamodule.custom_nodes_dataloader(calib_tune_nodes, batch_size=ctd_bs, sampler=sampler)

    cqd_bs = batch_size
    if batch_size == -1:
        cqd_bs = len(calib_qscore_nodes)
    calib_qscore_dl = datamodule.custom_nodes_dataloader(calib_qscore_nodes, batch_size=cqd_bs, sampler=sampler)


    t_bs = batch_size
    if batch_size == -1:
        t_bs = len(test_nodes)
        
    test_dl = datamodule.custom_nodes_dataloader(test_nodes, batch_size=t_bs, sampler=sampler)

    # first fit the cfgnn
    with utils.dl_affinity_setup(calib_tune_dl)():
        start = time.time()
        trainer.fit(cfgnn, calib_tune_dl)
        end = time.time()

    # next, compute the quantile
    with utils.dl_affinity_setup(calib_qscore_dl)():
        with torch.no_grad():
            trainer.test(cfgnn, calib_qscore_dl)
            scores = cfgnn.latest_test_results[SCORES_KEY]
            labels = cfgnn.latest_test_results[LABELS_KEY]

    label_scores = torch.gather(scores, 1, labels.unsqueeze(1))
    quantile = cfgnn.eval_score_fn.compute_quantile(label_scores, cfgnn.alpha)

    # next, compute test effectiveness
    with utils.dl_affinity_setup(test_dl)():
        with torch.no_grad():
            trainer.test(cfgnn, test_dl)
            test_scores = cfgnn.latest_test_results[SCORES_KEY]
            test_labels = cfgnn.latest_test_results[LABELS_KEY]

    pred_sets = PredSetTransformation(qhat=quantile).pipe_transform(test_scores)

    eff = compute_metric(conf_metric_names.efficiency.name, pred_sets, test_labels, cfgnn.alpha, None)
    return eff, end - start

def setup_fullbatch_baseline(args: RuntimeBenchmarkingConfig):
    args.lr = 1e-3
    args.batch_size = -1
    args.load_probs = False
    args.max_epochs = 1000
    args.train_fn = "tps"
    args.eval_fn = "aps"


def setup_our_baseline(args: RuntimeBenchmarkingConfig):
    args.lr = 5e-3
    args.batch_size = 64
    args.load_probs = True
    args.max_epochs = 20
    args.train_fn = "aps"
    args.eval_fn = "aps"


def main() -> None:
    args = pyr_a.parse(config_class=RuntimeBenchmarkingConfig)

    # first we compare batching improvements
    # use the default settings
    results = []

    # First, our method, with caching and batching
    method = "cache+batch"
    logging.info(f"Running {args.n_runs_per_expt} runs of {method}")
    for idx in range(args.n_runs_per_expt):
        setup_our_baseline(args)
        eff, runtime = run_and_time_conformal(args)
        results.append({"run_idx": idx, "method": method, "efficiency": eff, "runtime": runtime})

    # then their method
    method = "baseline"
    logging.info(f"Running {args.n_runs_per_expt} runs of {method}")
    for idx in range(args.n_runs_per_expt):
        setup_fullbatch_baseline(args)
        eff, runtime = run_and_time_conformal(args)
        results.append({"run_idx": idx, "method": method, "efficiency": eff, "runtime": runtime})
    
    method = "batching"
    logging.info(f"Running {args.n_runs_per_expt} runs of {method}")
    for idx in range(args.n_runs_per_expt):
        setup_our_baseline(args)
        args.load_probs = False
        eff, runtime = run_and_time_conformal(args)
        results.append({"run_idx": idx, "method": method, "efficiency": eff, "runtime": runtime})

    logging.info("Writing output file") 
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.benchmark_output_file, index=False)
    

if __name__ == "__main__":
    #DATASET=CiteSeer
    #BASE_JOB_ID=best_${DATASET}_split_0.2_0.2
    #BASE_OUTPUT_DIR=/data/avoirpp/avoirpp_best_base/${DATASET}/split/0.2_0.2/
    #python time_improvements_cfgnn.py --dataset ${DATASET} --base_job_id ${BASE_JOB_ID} --base_output_dir ${BASE_OUTPUT_DIR} --dataset_dir ./datasets --conformal_seed 0 --alpha 0.1 --n_runs_per_expt 5 --benchmark_output_file ./analysis/benchmarking/${DATASET}_0.2_0.2.csv
    main()
