from typing import List

import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback, TQDMProgressBar, ModelCheckpoint

from .custom_logger import CustomLogger

from .constants import LABELS_KEY, SCORES_KEY
from . import utils
from .data_module import DataModule
from .config  import ConfGNNConfig, ConfExptConfig
from .scores import CPScore
from .models import CFGNN

class CFGNNScore(CPScore):
    def __init__(
        self,
        conf_config: ConfExptConfig,
        datamodule: DataModule,
        confgnn_config: ConfGNNConfig,
        logger: CustomLogger,
    ):
        super().__init__(confgnn_config=confgnn_config)
        self.logger = logger
        self.alpha = conf_config.alpha


        self.trainable_model = CFGNN(
            config=confgnn_config,
            alpha=conf_config.alpha,
            num_epochs=conf_config.epochs,
            num_classes=datamodule.num_classes,
        )
        if not confgnn_config.load_probs:
            self.total_layers = (
                self.trainable_model.base_model.num_layers
                + self.trainable_model.confgnn.num_layers
            )
        else:
            self.total_layers = self.trainable_model.confgnn.num_layers

        callbacks: List[Callback] = [TQDMProgressBar(refresh_rate=100)]

        if confgnn_config.ckpt_dir is not None:
            best_callback = ModelCheckpoint(
                monitor="confgnn_val_eff",
                dirpath=confgnn_config.ckpt_dir,
                filename=f"confgnn-{confgnn_config.model}-{{epoch:02d}}-{{confgnn_val_eff:.2f}}",
                save_top_k=1,
                mode="min",
            )
            callbacks.append(best_callback)

        self.pt = utils.setup_trainer(conf_config, logger, callbacks=callbacks)

    def compute(self, dl, **kwargs):
        with utils.dl_affinity_setup(dl)():
            with torch.no_grad():
                self.pt.test(self.trainable_model, dataloaders=dl)
                scores, labels = (
                    self.trainable_model.latest_test_results[SCORES_KEY],
                    self.trainable_model.latest_test_results[LABELS_KEY],
                )

        return scores, labels

    def learn_params(self, calib_tune_dl, calib_qscore_dl):
        with utils.dl_affinity_setup(calib_tune_dl)():
            # first fit the model
            self.pt.fit(
                self.trainable_model,
                train_dataloaders=calib_tune_dl,
                val_dataloaders=calib_tune_dl,
                ckpt_path=None,
            )

        # determine qhat
        scores, labels = self.compute(calib_qscore_dl)
        label_scores = torch.gather(scores, 1, labels.unsqueeze(1)).squeeze()
        quantile = self.trainable_model.eval_score_fn.compute_quantile(
            label_scores, self.alpha
        )

        return quantile