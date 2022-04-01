from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset
import os
import pandas as pd

"""
NOTE: The dataset is a work in progress. For now, we only load X1 cell line data!!!
"""
class InputDataset(Dataset):

    def __init__(self, data_directory='./', modality_names=[('', '')]):
        super(InputDataset, self).__init__()
        self.data_path = data_directory

        # Below:
        self.modality_names = modality_names

        # initialize:
        self.data = []
        for name in modality_names:
            self.data.append(self.load_modality(name))

    def load_modality(self, name):
        modality_data = pd.read_csv(os.path.join(self.data_path, name, 'X1.bed'), sep='\t', header=None)
        modality_data = pd.read_csv(os.path.join(self.data_path, name, 'X1.bigwig'), sep='\t', header=None)
        return modality_data

    pd.read_csv('./X1.bed', sep='\t', header=None)

    def __len__(self):
        if self.data is not None:
            return self.data[0].shape[0]
        else:
            return 0

    def __getitem__(self, index):
