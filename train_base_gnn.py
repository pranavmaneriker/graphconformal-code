import logging
import os
import shutil
from contextlib import ExitStack

import graph_conformal.utils as utils
import pyrallis.argparsing as pyr_a
import torch
from graph_conformal.config import BaseExptConfig
from graph_conformal.custom_logger import CustomLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    args = pyr_a.parse(config_class=BaseExptConfig)
    # Callbacks
    # setup checkpointing
    ckpt_dir, ckpt_filename = utils.get_base_ckpt_dir_fname(
        args.output_dir, args.dataset, args.job_id
    )

    if not args.resume_from_checkpoint:
        # delete existing chekpoint dir if it exists
        logger.warning(
            f"Existing checkpoint for {args.dataset}/{args.job_id} will be overwritten."
        )
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
    else:
        logger.warning("Resuming from checkpoint")
        args = utils.load_basegnn_config_from_ckpt(ckpt_dir, args)

    # overwrite any existing config
    utils.output_basegnn_config(ckpt_dir, args)

    utils.set_seed_and_precision(args.seed)
    datamodule = utils.prepare_datamodule(args)
    datamodule.setup_sampler(args.base_gnn.layers, args.base_gnn.fanouts)

    # create logger and log expt hyperparams
    expt_logger = CustomLogger(args.logging_config)
    expt_logger.log_hyperparams(vars(args))

    model = utils.load_basegnn(ckpt_dir, args, datamodule)

    best_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    callbacks = [TQDMProgressBar(refresh_rate=100), best_callback]
    trainer = utils.setup_trainer(
        args,
        expt_logger,
        strategy="ddp" if args.use_ddp else "auto",
        callbacks=callbacks,
    )

    # fix cpu affinity issues
    # details: https://docs.dgl.ai/tutorials/cpu/cpu_best_practises.html
    # using contextlib.ExitStack from https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
    with ExitStack() as stack:
        train_dl, val_dl = utils.enter_cpu_cxs(
            datamodule,
            ["train_dataloader", "val_dataloader"],
            stack,
            datamodule.num_workers,
        )
        # TODO: ensure that this loads the training checkpoint if halted
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=None,
        )
        # TODO: Is actual eval on test required?
        # results = trainer.test(
        #    model=model,
        #    dataloaders=test_dl,
        #    ckpt_path=best_callback.best_model_path
        # ) #actual eval on test

    # run on all to get scores to use with alternative splits
    results = utils.run_basegnn_inference_alldl(
        model, trainer, best_callback.best_model_path, datamodule
    )
    if results is not None:
        utils.output_basegnn_results(args, results)
    else:
        logger.error("No results to output")
        raise ValueError("No results to output")


if __name__ == "__main__":
    # python train_base_gnn.py  --config_path="configs/base_gnn_default.yaml"
    main()
