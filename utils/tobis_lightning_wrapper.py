import numpy as np
import pytorch_lightning as pl
import torch

import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import statistics


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
        x = batch[0]
        y = batch[1]
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def train_dataloader(self):
        return DataLoader(self.datasets[0], batch_size=self.batch_size,
                          shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.datasets[1], batch_size=self.batch_size,
                          shuffle=False, drop_last=False, num_workers=8)

    """def test_dataloader(self):
        return DataLoader(self.datasets[2], batch_size=self.batch_size)"""
