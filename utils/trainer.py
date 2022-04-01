# ------------
# Author:       Philip Toma
# Description:  This file implements the training pipeline of the IFBID Model.
# Usage:        python3 trainer.py [--debug] [--epochs 10] [path]
#               where the []-brackets mean an entry is optional, but should be used.
#               For more info:      python3 trainer.py --help
# ------------

import pytorch_lightning as pl
import torch

from utils.lightning_wrapper import *
from utils.dataset import InputDataset

import argparse
from math import ceil
import csv

# Ensure Reproducability:
pl.seed_everything(2022, workers=True)

# Initialise model data. Set new_model=True to see how we train on the generalisation set
data = InputDataset()

model = Model()

# Initialize training-loss
loss = torch.nn.BCELoss()

# Initialise test_data.  Set new_model=True to see how we train on the generalisation set.
test_data = InputDataset()

# Initialise pl model and trainer
lightning_model = ModelWrapper(model_architecture=model, learning_rate=1e-3, loss=loss, dataset=data,
                               dataset_distr=[int(0.7*len(data)), ceil(len(data) - 0.7*len(data))], test_dataset=test_data,
                               batch_size=batch_size)

trainer = pl.Trainer(max_epochs=20, deterministic=True, reload_dataloaders_every_n_epochs=5)

# Train the model.
trainer.fit(lightning_model)

# Test the model.
trainer.test(lightning_model)

# get and record accuracy obtained from test.
# test_accuracy = lightning_model._model.test_accuracy

print(f'Save model under name: {name}{tmp}-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}')

with open(os.path.join(args.path, f'{name}{tmp}-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow([test_accuracy])

if not args.debug:
    os.makedirs(os.path.join(args.path, 'bias_classifiers'), exist_ok=True)
    torch.save(lightning_model._model.state_dict(),
               os.path.join(args.path, 'bias_classifiers',
                            f'{name}{tmp}-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}')
               )
