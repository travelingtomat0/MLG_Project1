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

import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import statistics


class ModelWrapper(pl.LightningModule):

    def __init__(self, model_architecture, learning_rate, loss, dataset=None, dataset_distr=None,
                 batch_size=1):
        super(ModelWrapper, self).__init__()
        self._model = model_architecture
        self.lr = learning_rate
        self.loss = loss
        if dataset is not None:
            self.dataset_split = random_split(dataset, dataset_distr)
        self.batch_size = batch_size

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def train_dataloader(self):
        return DataLoader(self.dataset_split[0], batch_size=self.batch_size,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_split[1], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_split[2], batch_size=self.batch_size)