import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from termcolor import colored
from typing import List, Tuple, Optional

# import sys
# sys.path.append(os.path.dirname(__file__))

from .utils import tqdm

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

    def calc_PAC(self, u1=0.1, u2=0.9, block_size=1_000, use_torch=True, device="gpu", **kwargs):
        """ """
        from . import PAC_score
        self.PAC = PAC_score.calc_PAC_block(self, u1=u1, u2=u2, use_torch=use_torch,
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

        min_opt_k = 5
        self.optimal_k = self.ks[np.argmin(self.method_values[min_opt_k:]) + min_opt_k]
        self.optimal_k_method = method

        return self.optimal_k


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
            fig.savefig(save_plot_path.format(k=set_k))


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
