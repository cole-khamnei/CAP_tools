import argparse
import os

import dill as pickle
import nibabel as nb
import numpy as np
import pandas as pd

# import os
# import sys
# sys.path.append(os.path.dirname(__file__))

from . import plots
from . import clustering
from . import surface_mapping as sfm

from .utils import color_str
from .utils import tqdm
from typing import List, Optional, Tuple

# ------------------------------------------------------------------- #
# --------------------   CAP pipeline functions  -------------------- #
# ------------------------------------------------------------------- #


def get_cifti_paths(cifti_argument: List[str]) -> List[str]:
    """ Reads in the inputs to main for --cifti/-c.
        Valid arguments are a .txt file with lines being individual ciftis, or glob of paths (.nii)

    """
    path_endings = [path.split(".", maxsplit=1)[1] for path in cifti_argument]
    file_ext = path_endings[0]

    assert all(ending == file_ext
               for ending in path_endings[1:]), "Multiple cifti input types not supported"
    assert all(ending in ["txt", "dtseries.nii", "ptseries.nii"]
               for ending in path_endings), f"invalid ending type {file_ext}"

    if file_ext == "txt":
        assert len(cifti_argument) == 1, "Multiple cifti path text files not supported."
        cifti_path_txt_file = cifti_argument[0]

        with open(cifti_path_txt_file, "r") as file:
            return file.read().strip().split()

    elif file_ext == "ptseries.nii":
        return cifti_argument

    elif file_ext == "dtseries.nii":
        return cifti_argument

    raise NotImplementedError("Control flow should not get here.")


def preprocess_cifti_array(raw_cifti_data_array: np.ndarray,
                           **preprocess_arguments) -> np.ndarray:
    """ """
    print(color_str("Preprocessing CIFTI data array: ...", "blue"), end="\r")
    if "arg" in preprocess_arguments:
        raise NotImplementedError
    else:
        pass

    print(color_str("Preprocessing CIFTI data array: Done", "blue"))
    return raw_cifti_data_array


def ISC_subset(cifti_array: np.ndarray,
               ROI_labels: list,
               isc_threshold: float = 0.15,
               pbar: bool = True,
               save_plots: bool = True,
               save_path: Optional[str] = None) -> np.ndarray:
    """ """
    isc_df = []

    pbar_kwargs = dict(desc=color_str("Calculating ISC Values", "blue"), colour="blue")
    ROI_iter = tqdm(ROI_labels, **pbar_kwargs) if pbar else ROI_labels

    # Loading ISC takes a long time
    from nltools.stats import isc
    for i, ROI in enumerate(ROI_iter):
        res = isc(cifti_array[:, :, i].T)
        res["roi"] = ROI
        isc_df.append(res)
    isc_df = pd.DataFrame(isc_df)
    isc_thres_df = isc_df.query(f"isc >= {isc_threshold}")

    # TODO: Add ISC thresholding Plots:

    assert len(isc_thres_df) > 0, f"ISC threshold = {isc_threshold} led to zero ROIs, lower threshold."
    return np.sort(isc_thres_df["roi"].values), isc_df


def create_dCAP_states(cifti_array: np.ndarray, CAP_labels: np.ndarray,
                       dtseries_paths: List[str], pbar: bool = True,
                            ) -> np.ndarray:
    """ """

    assert cifti_array.shape[0] == len(dtseries_paths), "Incorrect number of dtseries provided."
    assert CAP_labels.shape[0] == np.prod(cifti_array.shape[:2]), "CAP labels shape incorrect."

    N_ciftis, N_TRs, N_rois = cifti_array.shape
    DT_VOXEL_NUMBER = len(nb.load(dtseries_paths[0]).header.get_axis(1))

    CAP_labels = CAP_labels.reshape(N_ciftis, N_TRs)
    CAP_state_nums = np.sort(np.unique(CAP_labels))

    dCAP_states = {state_i: np.zeros(DT_VOXEL_NUMBER) for state_i in CAP_state_nums}
    dCAP_state_norms = {state_i: 0 for state_i in CAP_state_nums}

    if pbar:
        pbar = tqdm(total=N_ciftis, desc=color_str("Creating CAP State dscalars", "blue"), colour="blue")

    for dtseries_path, cifti_CAP_labels in zip(dtseries_paths, CAP_labels):
        dtseries_cifti = nb.load(dtseries_path)
        dtseries_data = dtseries_cifti.get_fdata()

        for state_i in CAP_state_nums:
            cifti_CAP_i_index = cifti_CAP_labels == state_i

            dCAP_states[state_i] += np.sum(dtseries_data[cifti_CAP_i_index, :], axis=0)
            dCAP_state_norms[state_i] += np.count_nonzero(cifti_CAP_i_index)

        if pbar:
            pbar.update(1)

    for state_i in dCAP_states.keys():
         dCAP_states[state_i] /=  dCAP_state_norms[state_i]

    return np.array([dCAP_states[i] for i in sorted(dCAP_states.keys())])


# ------------------------------------------------------------------- #
# --------------------            END            -------------------- #
# ------------------------------------------------------------------- #
