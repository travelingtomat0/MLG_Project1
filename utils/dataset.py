import numpy as np
from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset
import os
import pandas as pd

"""
NOTE: The dataset is a work in progress. For now, we only load X1 cell line data!!!
"""
class InputDataset(Dataset):

    def __init__(self, data_directory='./', window_size=200000, cell_line='X1', modality_names=[('', '')]):
        super(InputDataset, self).__init__()
        self.data_path = data_directory
        self.cell_line = cell_line
        # Below:
        self.modality_names = modality_names
        # Window size refers to offset in __one direction__! Ie. effective size is twice as large.
        self.window_size = window_size

        # ----------
        # Initialize
        # ----------
        # Load information
        self.train_info = pd.read_csv(os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_train_info.tsv'), sep='\t')

        # set length of dataset.
        self.length = self.train_info.shape[0]

        # Generate data per gene
        for gene in self.train_info['gene_name']:
            pass
        self.data = []
        for name in modality_names:
            self.data.append(self.load_modality(name))

    def load_modality(self, name):
        modality_data = pd.read_csv(os.path.join(self.data_path, name, 'X1.bed'), sep='\t', header=None)
        modality_data = pd.read_csv(os.path.join(self.data_path, name, 'X1.bigwig'), sep='\t', header=None)
        return modality_data

    def get_bed_vector(self, modality, chr, TSS_start, TSS_end):
        bed_file = pd.read_csv(os.path.join(self.data_path, modality, f'{self.cell_line}.bed'))
        # TODO: HOW ORIENTATION?
        resulting_peaks = np.zeros((2*self.window_size + (TSS_end - TSS_start)))
        peaks = bed_file.loc[(bed_file[0] == chr) &
                             (bed_file[1] > TSS_start - self.window_size) &
                             (bed_file[2] < TSS_end + self.window_size)]
        # one-hot encoding of the peaks...
        for peak in peaks:
            for i in range(0):
                pass
        bed_file = None

    pd.read_csv('./X1.bed', sep='\t', header=None)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # index+1 because 0 returns the header
        info = self.train_info[index:index+1]
        TSS = (self.train_info['TSS_start'][index], self.train_info['TSS_end'][index])
        gene_region = (self.train_info['gene_start'][index], self.train_info['gene_end'][index])
        chr = self.train_info['chr'][index]
