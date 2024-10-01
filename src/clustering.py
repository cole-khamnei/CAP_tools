import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from termcolor import colored
from typing import List, Tuple, Optional

import utils
from utils import tqdm

try:
    import torch
    # TODO: Change to work on CUDA and CPU as well
    assert torch.backends.mps.is_available()
    device = torch.device("mps")
    TORCH_AVAILABLE = True
    print(colored("Torch found, using as backend for CAP_state.ConsensusKMeans.", "yellow"))
except ModuleNotFoundError as e:
    TORCH_AVAILABLE = False
    print(colored("Torch package not found. Defaulting to numpy.", "yellow"))

# ------------------------------------------------------------------- #
# --------------------        Torch Utils        -------------------- #
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
# --------------------       Random Utils        -------------------- #
# ------------------------------------------------------------------- #


def create_sampled_set(size: int, p: float, iterations: int):
    """ """
    return [np.sort(np.random.choice(np.arange(size), size=int(np.round(size * p)), replace=False))
           for k in range(iterations)]


def kmeans_convergence_check(km):
    """ """
    assert km.n_iter_ < km.max_iter, f"Convergence failed (max_iter = {km.max_iter}"


# ------------------------------------------------------------------- #
# --------------------        Clustering         -------------------- #
# ------------------------------------------------------------------- #


class MultiKMeans:
    def __init__(self,k: int, n_reps: int = 100, max_iter=1_200,
                 p_events: float = 1.0, p_features: float = 1.0):
        """
        """
        self.k = k
        self.max_iter = max_iter
        self.n_reps = n_reps
        self.p_events = p_events # not used now
        self.p_features = p_features # not used now
        self.build()

    def build(self):
        """ """
        self.is_fit = False
        self.cm_built = False
        self.kmeans = [KMeans(n_clusters=self.k, max_iter=self.max_iter)
                       for n_i in range(self.n_reps)]


    def create_sample_indices(self, X: np.ndarray, event_groupings=None):
        """ """
        n_events, n_features = X.shape
        self.feature_sets = create_sampled_set(n_features, self.p_features, self.n_reps)

        self.event_groupings = event_groupings
        if event_groupings is None:
            self.event_sets = create_sampled_set(n_events, self.p_events, self.n_reps)
        else:
            unique_groups = np.sort(np.unique(event_groupings))

            self.group_sets = [unique_groups[gs] for gs in create_sampled_set(len(unique_groups), self.p_events, self.n_reps)]
            self.event_sets = [np.where(np.isin(event_groupings, gs))[0] for gs in self.group_sets]


    def fit(self, X, event_groupings=None, pbar=False, **kwargs):
        """ """
        self.create_sample_indices(X, event_groupings=event_groupings)

        self.labels = []
        kwargs["colour"] = "blue"
        kwargs["desc"] = colored(f"Fitting MultiKMeans (k={self.k})", "blue")
        km_iter = tqdm(self.kmeans, **kwargs) if pbar else self.kmeans
        for i, kmeans in enumerate(km_iter):
            kmeans.fit(X[self.event_sets[i]][:, self.feature_sets[i]]) # Index with feature and event set
            kmeans_convergence_check(kmeans)

            labels_i = np.full(X.shape[0], np.nan)
            labels_i[self.event_sets[i]] = kmeans.labels_
            self.labels.append(labels_i)

        self.labels = np.array(self.labels).astype(np.float32)
        self.is_fit = True
        return self

    def build_consensus_matrix(self, use_torch=True):
        """ No selection matrix / selection count normalization done
            as all entrys are clustered together
        """
        assert self.is_fit

        if use_torch and TORCH_AVAILABLE:
            return self.build_cm_torch()

        n_samples = len(self.kmeans[0].labels_)
        self.consensus_matrix = np.zeros((n_samples, n_samples))

        for i in range(self.n_reps):
            labels_i = self.kmeans[i].labels_
            label_matrix = np.tile(labels_i, (n_samples, 1))
            self.consensus_matrix += label_matrix == label_matrix.T

        self.consensus_matrix /= self.n_reps
        self.cm_built = True
        return self.consensus_matrix

    def build_cm_torch(self):
        """ Builds the consensus matrix using torch >2x speed up"""

        assert TORCH_AVAILABLE

        n_samples = len(self.kmeans[0].labels_)
        tcm = torch.zeros(n_samples, n_samples)

        for i in range(self.n_reps):
            labels_i = self.kmeans[i].labels_
            tlabels_i = torch.from_numpy(labels_i)
            label_matrix = torch.tile(tlabels_i, (n_samples, 1))
            tcm += label_matrix == label_matrix.T

        tcm = tcm / self.n_reps

        self.consensus_matrix = tcm.numpy()
        self.cm_built = True
        return self.consensus_matrix

    def calc_PAC(self, u1=0.1, u2=0.9, block_size=1_000, use_torch=True, device="gpu", **kwargs):
        """ """
        self.PAC = calc_PAC_block(self, u1=u1, u2=u2, use_torch=use_torch,
                                  block_size=block_size, device=device, **kwargs)

        return self.PAC


    def __repr__(self):
        s = f"MultiKMeans: K={self.k}  n_reps={self.n_reps}"
        s += f"  p_features={self.p_features}  p_events={self.p_events}"
        return s


class ConsensusKMeans:
    def __init__(self, kmax=None, ks=None, kmin=2, verbose: bool = True,
                 n_reps: int = 100, u1: float = 0.1, u2: float = 0.9,
                 **kmeans_kws):
        """ """

        if kmax is not None:
            ks = np.arange(kmin, kmax + 1)
        elif ks is None:
            raise ValueError("Either `kmax` or `ks` need to be provided.")

        self.ks = ks
        self.n_reps = n_reps
        self.kmeans_kws = kmeans_kws
        self.pac_kws=dict(u1=u1, u2=u2)
        self.verbose = verbose

        self.multikmeans = []
        self.is_fit = False

    def fit(self, X, pbar=False, event_groupings=None) -> int:
        """ """

        tqdm_kwargs = dict(desc=colored("CKM - Fitting KMeans", "cyan"), colour="cyan")
        ks_iter = tqdm(self.ks, **tqdm_kwargs) if pbar else self.ks
        for k in ks_iter:
            mkm_k = MultiKMeans(k=k, n_reps=self.n_reps,
                                **self.kmeans_kws).fit(X, pbar=pbar, event_groupings=event_groupings,
                                                       leave=False)
            self.multikmeans.append(mkm_k)

        self.is_fit = True
        return self

    def find_optimal_k(self, method="PAC", pbar=False, **method_kws):
        """ """
        assert self.is_fit
        self.method_values = []

        method_kws["pbar"] = pbar
        method_kws["pbar_kws"] = {"leave": False}
        if pbar:
            mkm_iter = tqdm(self.multikmeans, colour="cyan",
                            desc=colored(f"Finding optimal K using {method}", "cyan"))
        else:
            mkm_iter = self.multikmeans

        for mkm in mkm_iter:
            method_func = getattr(mkm, f"calc_{method}")
            self.method_values.append(method_func(**method_kws))

        self.optimal_k = self.ks[np.argmin(self.method_values[2:]) + 2]
        self.optimal_k_method = method

        return self.optimal_k


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
# --------------------    CAP State analysis     -------------------- #
# ------------------------------------------------------------------- #


def find_CAP_states(cifti_array: np.ndarray, ROI_labels: List[str], ROI_subset: Optional[list] = None,
                    seed: int = 0,
                    pbar: bool = True,
                    set_k = None,
                    cifti_sampling: bool = True,
                    kmax = 20,
                    n_reps = 40, #TODO add flags,
                    kmin = 3,
                    p_features = 1,
                    p_events = 0.8,
                    save_plot_path = None,
                    **CKM_params) -> Tuple[np.ndarray, np.ndarray]:
    """ """

    # TODO: Create arguments for all CKM params

    np.random.seed(seed)
    all_frames = np.vstack(cifti_array)

    n_ciftis, fMRI_length, n_parcels = cifti_array.shape
    if cifti_sampling:
        cifti_groupings = np.tile(np.arange(n_ciftis).reshape(-1, 1), (1, fMRI_length)).flatten()
    else:
        cifti_sampling = None

    if ROI_subset is None:
        cluster_array = all_frames
    else:
        ROI_subset_index = np.where(np.isin(ROI_labels, ROI_subset))[0]
        cluster_array = all_frames[:, ROI_subset_index]

    ncluster_array = cluster_array / np.sqrt(np.sum(cluster_array ** 2, axis=1)).reshape(-1, 1)

    if set_k is None:
        CKM = ConsensusKMeans(kmax=kmax, kmin=kmin, n_reps=n_reps, p_features=p_features, p_events=p_events)
        CKM.fit(ncluster_array, event_groupings=cifti_groupings, pbar=pbar)
        set_k = CKM.find_optimal_k(pbar=pbar)
        print(colored(f"Found optimal K={set_k}.", "yellow"))

        if save_plot_path:
            assert os.path.exists(os.path.dirname(save_plot_path)), os.path.dirname(save_plot_path)
            k_s = np.arange(kmin, kmax + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(k_s, CKM.method_values)
            ax.set_xlabel("K"), ax.set_ylabel(CKM.optimal_k_method)
            fig.savefig(save_plot_path)


    else:
        print(colored(f"Using provided K={set_k}.", "yellow"))
        # TODO: Create PAC minimum K selection plot

    np.random.seed(seed + 232)

    km = KMeans(n_clusters=set_k, max_iter=700, verbose=0, random_state=seed)
    km.fit(ncluster_array)

    CAP_states = []
    for cluster in np.arange(set_k):
        CAP_states.append(np.mean(all_frames[km.labels_ == cluster], axis=0))

    return CAP_states, km.labels_


# ------------------------------------------------------------------- #
# --------------------            END            -------------------- #
# ------------------------------------------------------------------- #
