import numpy as np
from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset
import os
import pandas as pd
import pyBigWig

"""
NOTE: The dataset is a work in progress. For now, we only load X1 cell line data!!!
"""
class InputDataset(Dataset):

    def __init__(self, data_directory='./', window_size=200000, cell_line='X1', modality_names=[]):
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

        # Data is fetched when indexing the dataset.
        # Implemented in __get_item__().
        pass

    def load_modality(self, name):
        modality_data = pd.read_csv(os.path.join(self.data_path, name, 'X1.bed'), sep='\t', header=None)
        modality_data = pd.read_csv(os.path.join(self.data_path, name, 'X1.bigwig'), sep='\t', header=None)
        return modality_data

    def get_bed_vector(self, modality, chr, start, end, TSS_start):
        bed_file = pd.read_csv(os.path.join(self.data_path, f'{modality}-bed/{self.cell_line}.bed'))
        # TODO: HOW ORIENTATION?
        resulting_peaks = np.zeros((2*self.window_size))
        peaks = bed_file.loc[(bed_file[0] == chr) &
                             (bed_file[1] >= start) &
                             (bed_file[2] <= end)]
        # one-hot encoding of the peaks...
        for i in range(len(peaks)):
            for j in range(peaks[i:i+1][1][0], peaks[i:i+1][2][0]):
                transformed_index = j - (TSS_start - self.window_size)
                resulting_peaks[transformed_index] = 1.0
        bed_file = None
        return resulting_peaks

    def get_bigwig_vector(self, modality, chr, start, end):
        try:
            bw = pyBigWig.open(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bigwig'))
        except:
            bw = pyBigWig.open(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bw'))
        resulting_vector = bw.values(chr, start, end)
        return np.array(resulting_vector)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # index+1 because 0 returns the header
        info = self.train_info[index:index+1]

        # Get the window
        lowest_location = info['TSS_start'][0] - self.window_size
        highest_location = info['TSS_end'][0] + self.window_size

        # Which cromosome is the gene located on?
        chr = info['chr'][0]

        # TODO: HOW TO USE STRAND +/- information?

        # Get binary input vector eg. methylation, acetylation etc. from .bed files.
        for m in self.modality_names:
            bed_vec = self.get_bed_vector(m, chr, lowest_location, highest_location, info['TSS_start'][0])
            bw_vec = self.get_bigwig_vector(self, m, chr, lowest_location, highest_location)

        # TODO: CONCATENATE VECTORS
        # TODO: Use sequence information?