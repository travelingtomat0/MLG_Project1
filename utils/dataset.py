import numpy as np
from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset
import os
import pandas as pd
import pyBigWig

"""
NOTE: The dataset is a work in progress. For now, we only load X1 cell line data!!!
"""

# db = {'T': 0, 'C': 1, 'A': 2, 'G': 3}

"""
@ USAGE INFORMATION:
Example how to use the dataset:
(1) Initialization
data = InputDataset(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X1'
                    , modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K9me3'])
(2) Retrieving gene-specific matrix (X) and gene_expression (y).
X, y = data[index]
--> len(data) retrieves the number of genes for which we have information.
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
        self.train_info = pd.read_csv(
            os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_train_info.tsv'), sep='\t'
        )

        self.y = pd.read_csv(
            os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_train_y.tsv'), sep='\t'
        )
        # set length of dataset.
        self.length = self.train_info.shape[0]

        # Data is fetched when indexing the dataset.
        # Implemented in __get_item__().
        pass

    def get_bed_vector(self, modality, chr, start, end, TSS_start):
        bed_file = pd.read_csv(os.path.join(self.data_path, f'{modality}-bed/{self.cell_line}.bed'), header=None,
                               sep='\t')
        # TODO: HOW ORIENTATION?
        resulting_peaks = np.zeros((2 * self.window_size))
        peaks = bed_file.loc[(bed_file[0] == chr) &
                             (bed_file[1] >= start) &
                             (bed_file[2] <= end)]
        # print(peaks)
        # print(peaks.shape)
        # one-hot encoding of the peaks...
        for i in range(len(peaks)):
            peak = peaks[i:i+1]
            # print(f'{peak[1].values[0]} {peak[2].values[0]}')
            # print(peak[2].values[0] - peak[1].values[0])
            for j in range(peak[1].values[0], peak[2].values[0]):
                # transformed_index = j - (TSS_start - self.window_size)
                transformed_index = j - start
                #print(transformed_index)
                resulting_peaks[transformed_index] = 1.0
        bed_file = None
        return resulting_peaks

    def get_bigwig_vector(self, modality, chr, start, end):
        if os.path.exists(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bigwig')):
            bw = pyBigWig.open(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bigwig'))
        else:
            bw = pyBigWig.open(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bw'))
        resulting_vector = bw.values(chr, start, end)
        return np.array(resulting_vector)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # index+1 because 0 returns the header
        info = self.train_info[index:index + 1]
        gene_name = info['gene_name'].values[0]
        gex = self.y.loc[self.y['gene_name'] == gene_name]['gex'].values[0]

        # Get the window
        lowest_location = info['TSS_start'].values[0] - self.window_size
        # highest_location = info['TSS_end'][0] + self.window_size
        highest_location = info['TSS_start'].values[0] + self.window_size

        # Which cromosome is the gene located on?
        chr = info['chr'].values[0]
        TSS_start = info['TSS_start'].values[0]
        # TODO: HOW TO USE STRAND +/- information?

        # Get binary input vector eg. methylation, acetylation etc. from .bed files.
        output = None
        for m in self.modality_names:
            if m == 'DNase':
                # Special case?
                if output is None:
                    output = self.get_bed_vector(m, chr, lowest_location, highest_location, TSS_start)
                    bw_vec = self.get_bigwig_vector(m, chr, lowest_location, highest_location)
                    output = np.vstack((output, bw_vec))
                else:
                    bed_vec = self.get_bed_vector(m, chr, lowest_location, highest_location, TSS_start)
                    bw_vec = self.get_bigwig_vector(m, chr, lowest_location, highest_location)
                    output = np.vstack((output, bed_vec, bw_vec))
            else:
                if output is None:
                    output = self.get_bed_vector(m, chr, lowest_location, highest_location, TSS_start)
                    bw_vec = self.get_bigwig_vector(m, chr, lowest_location, highest_location)
                    output = np.vstack((output, bw_vec))
                else:
                    bed_vec = self.get_bed_vector(m, chr, lowest_location, highest_location, TSS_start)
                    bw_vec = self.get_bigwig_vector(m, chr, lowest_location, highest_location)
                    output = np.vstack((output, bed_vec, bw_vec))
        # print(output)

        return output, gex
        # TODO: Use sequence information?
        # TODO: DNase information important at TSS or at gene-location? (the latter, right?)
