import argparse
import os

import dill as pickle
import nibabel as nb
import numpy as np
import pandas as pd

from . import plots
from . import clustering
from . import surface_mapping as sfm

from .utils import color_str
from tqdm.auto import tqdm
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------------# 
# ---------------            CAP Pipeline Functions            ---------------# 
# ----------------------------------------------------------------------------# 


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


def preprocess_cifti_array(raw_cifti_data_array: np.ndarray, crop=False,
                           **preprocess_arguments) -> np.ndarray:
    """ """
    print(color_str("Preprocessing CIFTI data array: ...", "blue"), end="\r")
    if "arg" in preprocess_arguments:
        raise NotImplementedError
    else:
        pass

    array_lengths = [len(obj) for obj in raw_cifti_data_array]

    if crop:
        print(f"Cropping {len(raw_cifti_data_array)} ciftis.")
        min_length = np.min(array_lengths) if isinstance(crop, bool) else crop
        if np.median(array_lengths) * 0.70 > np.min(array_lengths):
            print(f"WARNING: shortest array ({np.min(array_lengths)}) is less than 70% of median.")
            assert False

        processed_cifti_data_array = [arr[:min_length] for arr in raw_cifti_data_array]

    else:
        if any(array_lengths[0] != a_len for a_len in array_lengths):
            print("Warning: provided ciftis are different lengths (can use '--crop' to specify crop size')")

        processed_cifti_data_array = raw_cifti_data_array

    assert not any([np.any(np.isnan(ca)) for ca in processed_cifti_data_array])

    print(color_str("Preprocessing CIFTI data array: Done", "blue"))
    return processed_cifti_data_array


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

    assert len(isc_thres_df) > 0, f"ISC threshold = {isc_threshold} led to zero ROIs, lower threshold."
    return np.sort(isc_thres_df["roi"].values), isc_df


def create_dCAP_states(cifti_array: np.ndarray, CAP_labels_reshaped: list,
                       dtseries_paths: List[str], pbar: bool = True,
                       crop_slices: List[str] = [],
                            ) -> np.ndarray:
    """ """

    assert len(cifti_array) == len(dtseries_paths), "Incorrect number of dtseries provided."

    DT_VOXEL_NUMBER = len(nb.load(dtseries_paths[0]).header.get_axis(1))

    CAP_state_nums = np.sort(np.unique(np.hstack(CAP_labels_reshaped).ravel()))

    dCAP_states = {state_i: np.zeros(DT_VOXEL_NUMBER) for state_i in CAP_state_nums}
    dCAP_state_norms = {state_i: 0 for state_i in CAP_state_nums}

    if pbar:
        pbar = tqdm(total=len(cifti_array), desc=color_str("Creating CAP State dscalars", "blue"), colour="blue")

    for dtseries_path, cifti_CAP_labels in zip(dtseries_paths, CAP_labels_reshaped):
        dtseries_cifti = nb.load(dtseries_path)
        dtseries_data = dtseries_cifti.get_fdata()

        if len(crop_slices) > 0:
            c_index = np.ones(len(dtseries_data), dtype=bool)
            for cl in crop_slices:
                c_index[cl] = False
            dtseries_data = dtseries_data[c_index]


        for state_i in CAP_state_nums:
            cifti_CAP_i_index = cifti_CAP_labels == state_i
            crop_len = len(cifti_CAP_i_index)

            dCAP_states[state_i] += np.sum(dtseries_data[:crop_len][cifti_CAP_i_index, :], axis=0)
            dCAP_state_norms[state_i] += np.count_nonzero(cifti_CAP_i_index)

        if pbar:
            pbar.update(1)

    for state_i in dCAP_states.keys():
         dCAP_states[state_i] /=  dCAP_state_norms[state_i]

    return np.array([dCAP_states[i] for i in sorted(dCAP_states.keys())])


# ----------------------------------------------------------------------------# 
# --------------------                END                 --------------------# 
# ----------------------------------------------------------------------------#
