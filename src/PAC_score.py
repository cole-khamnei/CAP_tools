import os
import numpy as np

from termcolor import colored
from typing import List, Tuple, Optional

import sys
sys.path.append(os.path.dirname(__file__))

import utils
from utils import tqdm
try:
    import torch
    # TODO: Change to work on CUDA and CPU as well
    # TODO: Consider breaking out into separate script so torch isnt loaded always
    # TODO: Clean up
    TORCH_AVAILABLE = True
    print(colored("Torch found, using as backend for CAP_state.ConsensusKMeans.", "yellow"))
except ModuleNotFoundError as e:
    TORCH_AVAILABLE = False
    print(colored("Torch package not found. Defaulting to numpy.", "yellow"))


# ------------------------------------------------------------------- #
# --------------------        Torch Tools        -------------------- #
# ------------------------------------------------------------------- #


def get_torch_device(device="gpu"):
    """ gets torch device, defaulting to CUDA gpus"""
    device_name = device

    if device_name != "cpu":
        if torch.backends.mps.is_available():
            device_name = "mps"
        elif torch.cuda.is_available():
            device_name = "cuda"
        else:
            print(f"No torch device '{device_name}' available, defaulting to CPU.")
            device_name = "cpu"

    return torch.device(device_name)


# ------------------------------------------------------------------- #
# -------------------- Calculate PAC Optimization-------------------- #
# ------------------------------------------------------------------- #


def calc_PAC_block_CPU(mkm, u1=0.1, u2=0.9, block_size=1_000, pbar=True, use_torch=True, pbar_kws={}):
    """ """
    label_table = mkm.labels.astype(np.float32)
    n_samples = mkm.labels.shape[1]

    if use_torch and TORCH_AVAILABLE:
        label_table = torch.tensor(label_table)
        backend = torch
    else:
        backend = np

    SM_norm = backend.isnan(label_table).any()

    # split consensus matrix into nb blocks, then calculate PAC (or other metrics) on each block
    # iterate over blocks and combine metric

    n_blocks = int(np.ceil(n_samples / block_size))
    last_block_size = (n_samples % block_size) or block_size

    pac_s = []
    if pbar:
        pbar = tqdm(total= n_blocks * (n_blocks + 1) / 2,
                    desc=colored("Calculating PAC", "blue"), colour="blue", **pbar_kws)
    for i in range(n_blocks):

        i_block_size = last_block_size  if i == (n_blocks - 1) else block_size
        i_block_idx = slice(i * block_size, (i + 1) * block_size)

        for j in range(i + 1):
            j_block_size = last_block_size  if j == (n_blocks - 1) else block_size
            j_block_idx = slice(j * block_size, (j + 1) * block_size)

            c_ij_block = backend.zeros((i_block_size, j_block_size))

            if SM_norm:
                s_ij_block = backend.zeros((i_block_size, j_block_size))

            for rep_num in range(mkm.n_reps):
                i_block = label_table[rep_num, i_block_idx]
                j_block = label_table[rep_num, j_block_idx]
                a, b = backend.meshgrid(j_block, i_block, indexing="ij")
                c_ij_block += a == b

                if SM_norm:
                    s_ij_block += (~backend.isnan(a)) & (~backend.isnan(b))

            if SM_norm:
                c_ij_block = c_ij_block / s_ij_block
            else:
                c_ij_block /= mkm.n_reps

            counts = backend.sum(c_ij_block <= u2) - backend.sum(c_ij_block <= u1)
            pac_i = counts / (i_block_size * j_block_size)

            if i == j:
                pac_s.append(pac_i)
            else:
                pac_s.append(pac_i)
                pac_s.append(pac_i)
            if pbar:
                pbar.update(1)

    if use_torch:
        return float(torch.mean(torch.tensor(pac_s)))

    return np.mean(pac_s)


def calc_PAC_block(mkm, u1=0.1, u2=0.9, block_size=1_000, device="gpu", use_torch=True,
                   pbar=True, pbar_kws={}):
    """ """

    if TORCH_AVAILABLE:
        device = get_torch_device(device=device)

    if not use_torch or not TORCH_AVAILABLE or device.type == "cpu":
        return calc_PAC_block_CPU(mkm, u1=u1, u2=u2, block_size=block_size,
                                  use_torch=use_torch, pbar=pbar, pbar_kws=pbar_kws)

    label_table = torch.tensor(mkm.labels.T.astype(np.float32), device=device)
    n_samples = mkm.labels.shape[1]
    SM_norm = bool(torch.isnan(label_table).any())

    # split consensus matrix into nb blocks, then calculate PAC (or other metrics) on each block
    # iterate over blocks and combine metric

    n_blocks = int(np.ceil(n_samples / block_size))
    last_block_size = int(n_samples % block_size) or block_size

    pac_s = []
    if pbar:
        pbar = tqdm(total= n_blocks * (n_blocks + 1) / 2,
                    desc=colored("Calculating PAC", "blue"), colour="blue", **pbar_kws)

    for i in range(n_blocks):
        i_block_size = last_block_size  if i == (n_blocks - 1) else block_size
        i_block_idx = slice(i * block_size, (i + 1) * block_size)

        i_lt_slice = label_table[i_block_idx, :]
        i_tile = i_lt_slice.unsqueeze(1).repeat(1, block_size, 1)

        if SM_norm:
            i_tile_nan = ~ torch.isnan(i_tile)

        for j in range(n_blocks - i):
            j_block_size = last_block_size  if j == (n_blocks - 1) else block_size
            j_block_idx = slice(j * block_size, (j + 1) * block_size)

            j_lt_slice = label_table[j_block_idx, :]
            j_tile = torch.tile(j_lt_slice, (i_block_size, 1, 1))

            if j_block_size != i_tile.shape[0]:
                i_tile = i_lt_slice.unsqueeze(1).repeat(1, j_block_size, 1)
                i_tile_nan = ~ torch.isnan(i_tile)

            if SM_norm:
                j_tile_nan = ~ torch.isnan(j_tile)
                s_ij_block = (j_tile_nan & i_tile_nan).sum(axis=2)
                c_ij_block = (j_tile == i_tile).sum(axis=2) / s_ij_block
            else:
                c_ij_block = (j_tile == i_tile).sum(axis=2) / mkm.n_reps

            counts = torch.sum(c_ij_block <= u2) - torch.sum(c_ij_block <= u1)
            pac_i = counts / (i_block_size * j_block_size)

            pac_s.append(pac_i)
            if i != j:
                pac_s.append(pac_i) # doubles PAC values for non diagonal elements (C matrix is symmetric)

            if not isinstance(pbar, bool):
                pbar.update(1)

    if not isinstance(pbar, bool):
        pbar.close()

    return float(torch.mean(torch.tensor(pac_s)))


# ------------------------------------------------------------------- #
# --------------------      Consensus Matrix     -------------------- #
# ------------------------------------------------------------------- #


    # def build_consensus_matrix(self, use_torch=True):
    #     """ No selection matrix / selection count normalization done
    #         as all entrys are clustered together
    #     """
    #     assert self.is_fit

    #     if use_torch and TORCH_AVAILABLE:
    #         return self.build_cm_torch()

    #     n_samples = len(self.kmeans[0].labels_)
    #     self.consensus_matrix = np.zeros((n_samples, n_samples))

    #     for i in range(self.n_reps):
    #         labels_i = self.kmeans[i].labels_
    #         label_matrix = np.tile(labels_i, (n_samples, 1))
    #         self.consensus_matrix += label_matrix == label_matrix.T

    #     self.consensus_matrix /= self.n_reps
    #     self.cm_built = True
    #     return self.consensus_matrix

    # def build_cm_torch(self):
    #     """ Builds the consensus matrix using torch >2x speed up"""

    #     assert TORCH_AVAILABLE

    #     n_samples = len(self.kmeans[0].labels_)
    #     tcm = torch.zeros(n_samples, n_samples)

    #     for i in range(self.n_reps):
    #         labels_i = self.kmeans[i].labels_
    #         tlabels_i = torch.from_numpy(labels_i)
    #         label_matrix = torch.tile(tlabels_i, (n_samples, 1))
    #         tcm += label_matrix == label_matrix.T

    #     tcm = tcm / self.n_reps

    #     self.consensus_matrix = tcm.numpy()
    #     self.cm_built = True
    #     return self.consensus_matrix


# ------------------------------------------------------------------- #
# --------------------            END            -------------------- #
# ------------------------------------------------------------------- #
