from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import pandas as pd
import pyBigWig


class FastInputDataset(Dataset):
    """
    Dataset for input data.
    """

    def __init__(
        self,
        data_directory: Path,
        modality_names: List[str],
        window_size: int = 200000,
        cell_line: str = "X1",
        objective: str = "train",
        only_bw: bool = False,
    ):
        super().__init__()
        self.data_path = data_directory
        self.cell_line = cell_line
        self.objective = objective
        self.only_bw = only_bw
        self.modality_names = modality_names
        self.window_size = window_size

        info_df = pd.read_csv(
            data_directory / "CAGE-train" / f"{cell_line}_{objective}_info.tsv",
            sep="\t",
            header=0,
            index_col=None,
        )

        if objective != "test":
            y_df = pd.read_csv(
                data_directory / "CAGE-train" / f"{cell_line}_{objective}_y.tsv",
                sep="\t",
                header=0,
                index_col=None,
            )

            self.gene_df = pd.merge(info_df, y_df, on="gene_name")
        else:
            self.gene_df = info_df

        self.length = self.gene_df.shape[0]

        for modality in modality_names:
            print(f"Loading {modality} data...")
            bed = self._get_bed_df(modality)
            bigwig = self._get_bw_data(modality)
            bed_vectors = []
            bigwig_vectors = []
            for gene in tqdm(self.gene_df.itertuples(), total=self.length):
                bed_vectors.append(self._get_bed_vector(bed, gene.chr, gene.TSS_start))
                bigwig_vectors.append(self._get_bigwig_vector(bigwig, gene.chr, gene.TSS_start))
            self.gene_df[modality + "_bed"] = bed_vectors
            self.gene_df[modality + "_bigwig"] = bigwig_vectors

    def _get_bed_df(self, modality):
        return pd.read_csv(
            self.data_path / f"{modality}-bed/{self.cell_line}.bed",
            names=[
                "chrom",
                "chromStart",
                "chromEnd",
                "name",
                "score",
                "strand",
                "signalValue",
                "pValue",
                "qValue",
                "peak",
            ],
            sep="\t",
        )

    def _get_bw_data(self, modality):
        file_path = self.data_path / f"{modality}-bigwig"
        if os.path.exists(file_path / f"{self.cell_line}.bw"):
            return pyBigWig.open(str(file_path / f"{self.cell_line}.bw"))
        else:
            return pyBigWig.open(str(file_path / f"{self.cell_line}.bigwig"))

    def _get_bed_vector(self, bed_file, chr, TSS_start):
        """
        Returns a vector of self.window_size length centered at the TSS_start containing
        the score values of the modality peaks in the window.
        """

        start = TSS_start - self.window_size // 2
        end = TSS_start + self.window_size // 2

        peaks = bed_file.loc[
            (bed_file.chrom == chr)
            & (bed_file.chromEnd >= start)
            & (bed_file.chromStart <= end)
        ]

        peak_vector = np.zeros(self.window_size)
        for peak in peaks.itertuples():
            lo = max(0, peak.chromStart - start)
            hi = min(self.window_size, peak.chromEnd - start)
            peak_vector[lo:hi] = peak.score

        return peak_vector

    def _get_bigwig_vector(self, bigwig, chr, TSS_start):
        """
        Returns a vector of self.window_size length centered at the TSS_start containing
        the score values of the bigwig in the window.
        """

        start = TSS_start - self.window_size // 2
        end = TSS_start + self.window_size // 2

        q_start = max(start, 0)
        q_end = min(end, bigwig.chroms(chr))

        bw_vector = bigwig.values(chr, q_start, q_end)

        if start < q_start:
            bw_vector = np.concatenate((np.zeros(q_start - start), bw_vector))
        if end > q_end:
            bw_vector = np.concatenate((bw_vector, np.zeros(end - q_end)))

        return np.array(bw_vector)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # TODO: additional normalization?
        # TODO: strand information?
        # TODO: Use sequence information?

        gene = self.gene_df.iloc[index]

        gex = gene.gex

        # Get binary input vector eg. methylation, acetylation etc. from .bed files.
        features = []

        for modality in self.modality_names:
            features.append(gene[modality + "_bed"])
            features.append(gene[modality + "_bigwig"])
        output = np.vstack(features)

        return torch.from_numpy(output).float(), gex


