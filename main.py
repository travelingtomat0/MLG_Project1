from pathlib import Path
import pickle

import numpy as np
import pytorch_lightning as pl
import torch

from models.neural_nets import *
from utils.tobis_lightning_wrapper import ModelWrapper
from utils.fastdataset import FastInputDataset as InputDataset


def main():
    """
    Main function to run.
    """

    ###########################################################################
    # Setup
    ###########################################################################

    project_root = Path()

    data_path = project_root / "ML4G_Project_1_Data"

    # Modalities chosen by looking at the
    modalities = ["DNase"  ]#  , "H3K27ac", "H3K4me3", "H3K4me1", "H3K36me3"]
    window_size = 20000

    ###########################################################################
    # Load data
    ###########################################################################

    try:
        train_data_X1 = pickle.load(open("X1_train_data.pkl", "rb"))
    except FileNotFoundError:
        train_data_X1 = InputDataset(
            data_directory=data_path,
            cell_line="X1",
            objective="train",
            modality_names=modalities,
            window_size=window_size,
        )
        pickle.dump(train_data_X1, open("X1_train_data.pkl", "wb"))

    try:
        val_data_X1 = pickle.load(open("X1_val_data.pkl", "rb"))
    except FileNotFoundError:
        val_data_X1 = InputDataset(
            data_directory=data_path,
            cell_line="X1",
            objective="val",
            modality_names=modalities,
            window_size=window_size,
        )
        pickle.dump(val_data_X1, open("X1_val_data.pkl", "wb"))

    # train_data_X2 = InputDataset(
    #     data_directory=data_path,
    #     cell_line="X2",
    #     objective="train",
    #     modality_names=modalities,
    #     window_size=window_size,
    # )
    #
    # val_data_X2 = InputDataset(
    #     data_directory=data_path,
    #     cell_line="X2",
    #     objective="val",
    #     modality_names=modalities,
    #     window_size=window_size,
    # )
    #
    # test_data_X3 = InputDataset(
    #     data_directory=data_path,
    #     cell_line="X3",
    #     objective="test",
    #     modality_names=modalities,
    #     window_size=window_size,
    # )

    ###########################################################################
    # Define model
    ###########################################################################

    model = ConvolutionalModel(c=2*len(modalities))

    lightning_model = ModelWrapper(
        model_architecture=model,
        learning_rate=1e-3,
        loss=nn.MSELoss(),
        datasets=[train_data_X1, val_data_X1],
        batch_size=64,
    )

    trainer = pl.Trainer(
        max_epochs=10, deterministic=True, reload_dataloaders_every_n_epochs=5,
        num_sanity_val_steps=0,
    )

    ###########################################################################
    # Train model
    ###########################################################################

    if torch.cuda.is_available():
        trainer.fit(lightning_model, accelerator="gpu")
    else:
        trainer.fit(lightning_model)


    ###########################################################################
    # Make predictions
    ###########################################################################

    print("Hello")

    # TODO:
    # Using the model trained in WP 1.2, make predictions on the test data (chr 1 of cell line X3).
    # Store predictions in a variable called "pred" which is a numpy array.

    pred = None
    # ---------------------------INSERT CODE HERE---------------------------

    # ----------------------------------------------------------------------

    # Check if "pred" meets the specified constrains
    assert isinstance(pred, np.ndarray), "Prediction array must be a numpy array"
    assert np.issubdtype(pred.dtype, np.number), "Prediction array must be numeric"
    assert pred.shape[0] == len(
        test_genes
    ), "Each gene should have a unique predicted expression"

    #%% md

    #### Store Predictions in the Required Format

    #%%

    # Store predictions in a ZIP.
    # Upload this zip on the project website under "Your submission".
    # Zip this notebook along with the conda environment (and README, optional) and upload this under "Your code".

    save_dir = "path/to/save/output/file"  # TODO
    file_name = "gex_predicted.csv"  # PLEASE DO NOT CHANGE THIS
    zip_name = "LastName_FirstName_Project1.zip"  # TODO
    save_path = f"{save_dir}/{zip_name}"
    compression_options = dict(method="zip", archive_name=file_name)

    test_genes["gex_predicted"] = pred.tolist()
    test_genes[["gene_name", "gex_predicted"]].to_csv(
        save_path, compression=compression_options
    )

    #%% md
    # Playground
    #%%

    x, y = train_data_X1.__getitem__(10)
    x.shape

    #%%

    model(x)
    #%%

    y

    #%%

    len(train_data_X1)
    #%%

    for i in range(100):
        x, y = train_data_X1.__getitem__(i)
        print(f"{y:.2f} - {model(x).detach().numpy()}")

    #%%


if __name__ == "__main__":
    main()
