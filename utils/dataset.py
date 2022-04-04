import numpy as np
import torch

from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset
from torch.nn import AvgPool1d
import os
import pandas as pd
import pyBigWig
from tqdm import tqdm
import pickle

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

    def __init__(self, data_directory='./', window_size=200000, cell_line='X1', objective="train",
                 modality_names=[], only_bw=False, dim_reduction=None):
        super(InputDataset, self).__init__()
        self.data_path = data_directory
        self.cell_line = cell_line
        self.objective = objective
        self.only_bw = only_bw
        # Below:
        self.modality_names = modality_names
        # Window size refers to offset in __one direction__! Ie. effective size is twice as large.
        self.window_size = window_size
        # self.dim_reduction = dim_reduction(kernel_size=10)
        self.dim_reduction = AvgPool1d(kernel_size=10)

        # ----------
        # Initialize
        # ----------
        # Load information
        self.train_info = pd.read_csv(
            os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_{objective}_info.tsv'), sep='\t'
        )

        self.y = pd.read_csv(
            os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_{objective}_y.tsv'), sep='\t'
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
        try:
            resulting_vector = bw.values(chr, start, end)
        except:
            # print(end)
            # print(start)
            # print(bw.chroms(chr))
            # Two options:
            # (1) Prepend & Append 0s depending on case
            # (2) change view on the regions --> For this, we need to use these cases in bed-file embedding too!!
            if end > bw.chroms(chr):
                # resulting_vector = bw.values(chr, start - abs(bw.chroms(chr) - end), bw.chroms(chr))
                # APPEND
                resulting_vector = bw.values(chr, start, bw.chroms(chr))
                resulting_vector.extend([0.0 for k in range(abs(bw.chroms(chr) - end))])
            elif start < 0:
                # resulting_vector = bw.values(chr, 0, end+abs(start))
                # PREPEND:
                resulting_vector = [0.0 for k in range(abs(start))]
                resulting_vector.extend(bw.values(chr, 0, end))
            else:
                return np.zeros((2*self.window_size, ))
            # print(len(resulting_vector))
        return np.array(resulting_vector)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # index+1 because 0 returns the header
        info = self.train_info[index:index + 1]
        gene_name = info['gene_name'].values[0]
        gex = self.y.loc[self.y['gene_name'] == gene_name]['gex'].values[0]

        # Use gene length for normalization?
        """gene_length = info['gene_end'].values[0] - info['gene_start'].values[0]
        print(f'GEX before Norm: {gex}')
        gex = gex / gene_length
        print(f'GEX after Norm: {gex}')"""
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
            bed_vec = self.get_bed_vector(m, chr, lowest_location, highest_location, TSS_start)
            bw_vec = self.get_bigwig_vector(m, chr, lowest_location, highest_location)
            """if self.dim_reduction is not None:
                avg = AvgPool1d(kernel_size=10)
                print(avg(torch.from_numpy(bed_vec).unsqueeze(0)))
                bed_vec = avg(torch.from_numpy(bed_vec))
                bw_vec = avg(torch.from_numpy(bw_vec))"""
            if output is None:
                if self.only_bw:
                    output = np.multiply(bed_vec, bw_vec)
                else:
                    output = np.vstack((bed_vec, bw_vec))
            else:
                if self.only_bw:
                    output = np.vstack((output, np.multiply(bed_vec, bw_vec)))
                else:
                    output = np.vstack((output, bed_vec, bw_vec))

        return torch.from_numpy(output).float(), gex
        # TODO: Use sequence information?
        # TODO: DNase information important at TSS or at gene-location? (the latter, right?)


class Bigwig_Matrix_Builder:

    def __init__(self, data_directory='./', window_size=200000, cell_line='X1', objective="train",
                 modality_names=[], dim_reduction=False, num_bins=100, type='combined',
                 val=False, overwrite=False):
        self.cell_line = cell_line
        self.data_path = data_directory
        self.window_size = window_size
        self.objective = objective

        # retrieve info-file
        self.train_info = pd.read_csv(
            os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_{objective}_info.tsv'), sep='\t'
        )

        if dim_reduction is True:
            bin_size = 2*window_size // num_bins
            avg = AvgPool1d(kernel_size=bin_size)

        for modality in modality_names:
            # open bigwig file
            if os.path.exists(os.path.join(self.data_path, f'{cell_line}-{modality}-{type}-matrix-{num_bins}-{window_size}.idx')) \
                    and not overwrite and not val:
                with open(os.path.join(self.data_path, f'{cell_line}-{modality}-{type}-matrix-{num_bins}-{window_size}.idx'), 'rb') as f:
                    matrix = pickle.load(f)
                    print(f"{modality} has been loaded from disk.")
                    pass
            if val:
                cell_line = self.cell_line
                self.train_info = pd.read_csv(
                    os.path.join(data_directory, 'CAGE-train', 'CAGE-train', f'{cell_line}_val_info.tsv'),
                    sep='\t'
                )
                cell_line = cell_line + "-val"
            if os.path.exists(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bigwig')):
                bw = pyBigWig.open(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bigwig'))
            else:
                bw = pyBigWig.open(os.path.join(self.data_path, f'{modality}-bigwig/{self.cell_line}.bw'))

            # open .bed file
            bed_file = pd.read_csv(os.path.join(self.data_path, f'{modality}-bed/{self.cell_line}.bed'), header=None,
                                   sep='\t')

            tmp = []
            matrix = None

            for i in tqdm(range(len(self.train_info)), desc=f"Loading {modality} Matrix"):
                info = self.train_info[i: i+1]
                gene_name = info['gene_name'].values[0]
                # Get the window
                lowest_location = info['TSS_start'].values[0] - self.window_size
                highest_location = info['TSS_start'].values[0] + self.window_size
                chr = info['chr'].values[0]
                TSS_start = info['TSS_start'].values[0]
                # TODO: STRAND +/-
                """
                Negative-strand-coordinate-qStart = qSize - qEnd
                Negative-strand-coordinate-qEnd   = qSize - qStart
                """
                """if info['strand'].values[0] == '-':
                    qsize = bw.chroms(chr)
                    new_end = qsize - info['TSS_start'].values[0]
                    new_start = qsize - info['TSS_end'].values[0]"""
                """bed_vec = self.get_bed_vector(bed_file, chr, lowest_location, highest_location, TSS_start)
                bw_vec = self.get_bigwig_vector(bw, chr, lowest_location, highest_location)"""
                # dim_reduction can for instance be nn.avgpool1d
                """if dim_reduction is not None:
                    bed_vec = avg(torch.from_numpy(bed_vec).unsqueeze(0)).float()
                    bw_vec = avg(torch.from_numpy(bw_vec).unsqueeze(0)).float()"""

                if type == 'combined':
                    bed_vec = self.get_bed_vector(bed_file, chr, lowest_location, highest_location, TSS_start)
                    bw_vec = self.get_bigwig_vector(bw, chr, lowest_location, highest_location)
                    if dim_reduction is not None:
                        bed_vec = avg(torch.from_numpy(bed_vec).unsqueeze(0)).float()
                        bw_vec = avg(torch.from_numpy(bw_vec).unsqueeze(0)).float()
                    tmp.append(np.multiply(bed_vec, bw_vec))
                elif type == 'bw':
                    bw_vec = self.get_bigwig_vector(bw, chr, lowest_location, highest_location)
                    if dim_reduction is not None:
                        bw_vec = avg(torch.from_numpy(bw_vec).unsqueeze(0)).float()
                    tmp.append(bw_vec)
                else:
                    print(f'Problem parameter type={type}')

                if i % 100 == 99 and matrix is None:
                    matrix = np.vstack(tmp)
                    tmp = []
                elif i % 100 == 99:
                    matrix = np.vstack((matrix, np.vstack(tmp)))
                    #print(matrix.shape)
                    tmp = []
            # !!Add last elements!!
            if len(tmp) > 0:
                matrix = np.vstack((matrix, np.vstack(tmp)))
            print(matrix.shape)

            with open(os.path.join(self.data_path, f'{cell_line}-{modality}-{type}-matrix-{num_bins}-{window_size}.idx'), "wb") as f:
                print(f'Dumping {modality} with pickle.')
                pickle.dump(matrix, f)

    def get_bed_vector(self, bed_file, chr, start, end, TSS_start):
        # TODO: HOW ORIENTATION?
        resulting_peaks = np.zeros((2 * self.window_size))
        peaks = bed_file.loc[(bed_file[0] == chr) &
                             (bed_file[1] >= start) &
                             (bed_file[2] <= end)]
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

    def get_bigwig_vector(self, bw, chr, start, end):
        try:
            resulting_vector = bw.values(chr, start, end)
        except:
            # print(end)
            # print(start)
            # print(bw.chroms(chr))
            # Two options:
            # (1) Prepend & Append 0s depending on case
            # (2) change view on the regions --> For this, we need to use these cases in bed-file embedding too!!
            if end > bw.chroms(chr):
                # resulting_vector = bw.values(chr, start - abs(bw.chroms(chr) - end), bw.chroms(chr))
                # APPEND
                resulting_vector = bw.values(chr, start, bw.chroms(chr))
                resulting_vector.extend([0.0 for k in range(abs(bw.chroms(chr) - end))])
            elif start < 0:
                # resulting_vector = bw.values(chr, 0, end+abs(start))
                # PREPEND:
                resulting_vector = [0.0 for k in range(abs(start))]
                resulting_vector.extend(bw.values(chr, 0, end))
            else:
                return np.zeros((2*self.window_size, ))
            # print(len(resulting_vector))
        return np.array(resulting_vector)

"""
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X1', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=100, type='bw')
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X1', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=400, type='bw')
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X1', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=100)
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X1', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=400)
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X2', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=100, type='bw')
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X2', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=400, type='bw')
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X2', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=100)
data = Bigwig_Matrix_Builder(data_directory='/home/phil/Downloads/ML4G_Project_1_Data', cell_line='X2', modality_names=['DNase', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K36me3'], window_size=20000, dim_reduction=True, num_bins=400)"""

class RegressionData:

    def __init__(self, data_path="./", cell_line="X1", modalities=[], type='combined', num_bins=400, window_size=20000,
                 objective='train'):
        if objective != 'test':
            self.y = pd.read_csv(os.path.join(data_path, 'CAGE-train', 'CAGE-train', f'{cell_line}_{objective}_y.tsv'), sep='\t')
            try:
                self.y = self.y[:]['gex'].values
            except:
                self.y = self.y[0:-10]['gex'].values

        self.matrix = []
        for modality in modalities:
            with open(os.path.join(data_path, f'{cell_line}-{modality}-{type}-matrix-{num_bins}-{window_size}.idx'), 'rb') as f:
                m = pickle.load(f)
                self.matrix.append(m)

        self.matrix = np.hstack(self.matrix)
        print(f'Number of NaNs in matrix: {np.count_nonzero(np.isnan(self.matrix))}')
        self.matrix = np.nan_to_num(self.matrix)

        if objective == 'test':
            return

        self.val_matrix = []
        for modality in modalities:
            with open(os.path.join(data_path, f'{cell_line}-val-{modality}-{type}-matrix-{num_bins}-{window_size}.idx'), 'rb') as f:
                m = pickle.load(f)
                self.val_matrix.append(m)

        self.val_matrix = np.hstack(self.val_matrix)
        print(f'Number of NaNs in matrix: {np.count_nonzero(np.isnan(self.val_matrix))}')
        self.val_matrix = np.nan_to_num(self.val_matrix)
        self.y_val = pd.read_csv(
            os.path.join(data_path, 'CAGE-train', 'CAGE-train', f'{cell_line}_val_y.tsv'), sep='\t'
        )['gex'].values

# from scipy.stats import spearmanr
# clf = linear_model.Lasso(alpha=1000)
# clf.fit(r.matrix, r.y)
# p = clf.predict(r.val_matrix)
# spearmanr(clf.predict(r.val_matrix)-lol, r.y_val)
