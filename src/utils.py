import os
import shutil
import gc
import subprocess

import nibabel as nb
import numpy as np

from termcolor import colored
from typing import List, Tuple, Optional

from tqdm.auto import tqdm

# ----------------------------------------------------------------------------# 
# --------------------              Helpers               --------------------# 
# ----------------------------------------------------------------------------# 

def get_directory_size(path):
    result = subprocess.run(['du', path], capture_output=True, text=True)
    return int(result.stdout.split()[0]) / 2 ** 21


def cache_tmp_path(path, use_cache=True, write_cache=True):
    """ """
    if not isinstance(path, str):
        return [cache_tmp_path(path_i, use_cache=use_cache, write_cache=write_cache)
                for path_i in tqdm(path, desc="Caching paths in ~/_tmp")]

    if not use_cache:
        return path

    # TODO: builtin hash is non-deterministic, use hashlib or z-adler if want hashed option
    # tmp_path = f"/tmp/cifti_H{hash(path)}.{path.split('.', maxsplit=1)[1]}"

    tmp_path = "~/_tmp/" + path.split("/data/data", maxsplit=1)[1].lstrip("1234567/").replace("/", "--")
    tmp_path = os.path.expanduser(tmp_path)

    tmp_size = get_directory_size(os.path.expanduser("~/_tmp"))

    if not os.path.exists(tmp_path):
        if write_cache and tmp_size < 500:
            shutil.copyfile(path, tmp_path)
            gc.collect()
        else:
            tmp_path = path

    return tmp_path


def is_interactive() -> bool:
    import __main__ as main
    return not hasattr(main, '__file__')


def color_str(string: str, color: str) -> str:
    """ """
    return string if is_interactive() else colored(string, color)


def cifti_map(rois, roi_values, template_cifti, fill_value=np.nan):
    """ """

    if rois is not None:
        assert roi_values.shape[-1] <= len(rois)

    pax = template_cifti.header.get_axis(1)
    prefix_shape = roi_values.shape[:-1] if roi_values.ndim > 1 else ()    
    lh_values = np.full(shape=(pax.nvertices["CIFTI_STRUCTURE_CORTEX_LEFT"], *prefix_shape), fill_value=fill_value)
    rh_values = np.full(shape=(pax.nvertices["CIFTI_STRUCTURE_CORTEX_RIGHT"], *prefix_shape), fill_value=fill_value)

    if roi_values.ndim > 1:
        roi_values = roi_values.T
    # create plot values dict from template cifti
    if isinstance(pax, nb.cifti2.ParcelsAxis):
        for roi_value, roi in zip(roi_values, rois):
            _, kld = pax[roi]
            lh_values[kld.get("CIFTI_STRUCTURE_CORTEX_LEFT", [])] = roi_value
            rh_values[kld.get("CIFTI_STRUCTURE_CORTEX_RIGHT", [])] = roi_value

    elif isinstance(pax, nb.cifti2.BrainModelAxis):
        slice_LUT = {structure: sl for structure, sl,_  in pax.iter_structures()}
        lh_indices, rh_indices = [], []

        for i in range(len(pax)):
            _, ind, structure = pax[i]
            if "CORTEX_LEFT" in structure:
                lh_indices.append(ind)
            if "CORTEX_RIGHT" in structure:
                rh_indices.append(ind)

        lh_indices = np.array(lh_indices)
        rh_indices = np.array(rh_indices)

        lh_values[lh_indices] = roi_values[slice_LUT["CIFTI_STRUCTURE_CORTEX_LEFT"]]
        rh_values[rh_indices] = roi_values[slice_LUT["CIFTI_STRUCTURE_CORTEX_RIGHT"]]

    return {"left": np.moveaxis(lh_values, 0, -1), "right": np.moveaxis(rh_values, 0, -1)}


def ROI_subset_to_dlabel(ROI_subset: list, template_cifti, save_path: str, isc_threshold: float) -> None:
    """ """
    ROI_subset_values = cifti_map(ROI_subset, ROI_subset, template_cifti, fill_value="???")
    sfm.write_labels_to_dlabel(ROI_subset_values, save_path, label_name=f"ISC_subset_{isc_threshold}")


def write_CAP_scalars(CAP_states: np.ndarray, save_path: str,
                       cifti=None, cifti_path=None) -> None:
    """ """
    assert save_path.endswith((".pscalar.nii", ".dscalar.nii"))
    assert not ((cifti is None) and (cifti_path is None)), "Must provide either 'cifti' or 'cifti_path'"
    file_ext = save_path.split(".", maxsplit=1)[1]
    save_path = save_path.format(k=len(CAP_states))

    if cifti is None:
        cifti = nb.load(cifti_path)

    print(color_str(f"Writing CAP {file_ext}: ... ", "blue"), end="\r")

    brain_axis = cifti.header.get_axis(1)
    scalar_axis = nb.cifti2.ScalarAxis([f"CAP_state_{i + 1}" for i in range(len(CAP_states))])
    scalar_header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    scalar_cifti = nb.Cifti2Image(np.array(CAP_states), header=scalar_header)
    scalar_cifti.to_filename(save_path)

    print(color_str(f"Writing CAP {file_ext}: Done ", "blue"))


def load_cifti_arrays(cifti_paths: List[str], pbar: bool = True) -> Tuple[np.ndarray, List[str]]:
    """ """

    if pbar:
        file_ext = cifti_paths[0].split('.', maxsplit=1)[1]
        cifti_iter = tqdm(cifti_paths, colour="blue",
                          desc=color_str(f"Loading CIFTIs ({file_ext})", "blue"))
    else:
        cifti_iter = cifti_paths

    raw_cifti_data_array = []
    for cifti_path in cifti_iter:
        cifti = nb.load(cifti_path)
        raw_cifti_data_array.append(cifti.get_fdata())

    array_lengths = [len(obj) for obj in raw_cifti_data_array]

    # TODO: add error handling for short arrays
    if any(array_lengths[0] != a_len for a_len in array_lengths):
        print("Arrays of different lengths: cropping to minimum size.")
        min_length = np.min(array_lengths)
        if np.median(array_lengths) * 0.70 > np.min(array_lengths):
            print(f"WARNING: shortest array ({np.min(array_lengths)}) is less than 70% of median.")
            assert False

        raw_cifti_data_array = [arr[:min_length] for arr in raw_cifti_data_array]

    raw_cifti_data_array = np.array(raw_cifti_data_array)
    ROI_labels = [label for label, _, index in cifti.header.get_axis(1)]

    return raw_cifti_data_array, ROI_labels


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
