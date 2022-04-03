# ------------
# Author:       Philip Toma
# Description:  This file implements the lightning model, which takes a Bias Detector Model architecture
#               and then defines processes required for training and testing of the model.
# Usage:        lightning_model = ModelWrapper(model_architecture=classifier, learning_rate=1e-3, loss=some_loss,
#                                dataset=some_dataset,
#                                dataset_distr=[int(0.5*len(some_dataset)), int(0.25*len(some_dataset)),
#                                int(0.25*len(some_dataset))],
#                                batch_size=batch_size)
# ------------

import numpy as np
import pytorch_lightning as pl
import torch
import math

import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import statistics
from scipy.stats import spearmanr



class ModelWrapper(pl.LightningModule):

    def __init__(self, model_architecture, learning_rate, loss, datasets=None,
                 batch_size=1):
        super(ModelWrapper, self).__init__()
        self._model = model_architecture
        self.lr = learning_rate
        self.loss = loss
        if datasets is None:
            print("ERROR! No DATASET. NEED TO LOAD MANUALLY.")
        self.datasets = datasets
        self.batch_size = batch_size

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch[0][0]
        y = batch[1]
        y_hat = self._model(x)
        print(y)
        print(y_hat)
        return torch.log(self.loss(y_hat.float(), y.float()) + 1)

    def validation_step(self, batch, batch_idx):
        x = batch[0][0]
        y = batch[1]
        y_hat = self._model(x)
        # return self.loss(y_hat.float(), y.float())
        return torch.cat((y.float(), y_hat.float()))

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.stack(validation_step_outputs)
        print(spearmanr(all_preds[:, 0], all_preds[:, 1]))

    def spearman_corr(self, vec_1, vec_2):
        pass

    def train_dataloader(self):
        return DataLoader(self.datasets[0], batch_size=self.batch_size,
                          shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.datasets[1], batch_size=self.batch_size, num_workers=8)

    """def test_dataloader(self):
        return DataLoader(self.datasets[2], batch_size=self.batch_size)"""
