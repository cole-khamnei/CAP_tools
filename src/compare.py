import nibabel as nb
import numpy as np
import multiprocess as mp

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


import neuromaps as nm
from neuromaps import images, nulls, stats

# ----------------------------------------------------------------------------# 
# --------------------             Constants              --------------------# 
# ----------------------------------------------------------------------------# 

CORTEX_LEFT_LABEL = "CIFTI_STRUCTURE_CORTEX_LEFT"
CORTEX_RIGHT_LABEL = "CIFTI_STRUCTURE_CORTEX_RIGHT"

# ----------------------------------------------------------------------------# 
# --------------------           Cifti Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def load_dscalar_as_flat_gifti(cifti_path, n_vertex=32_492):
    """ """
    assert cifti_path.endswith(".dscalar.nii")

    cifti = nb.load(cifti_path)
    data = cifti.get_fdata()

    flat_data = np.zeros((data.shape[0], n_vertex * 2))
    for bm in cifti.header.get_index_map(1).brain_models:
        if bm.brain_structure not in [CORTEX_LEFT_LABEL, CORTEX_RIGHT_LABEL]:
            continue

        assert bm.surface_number_of_vertices == n_vertex

        bm_data = data[:, bm.index_offset:bm.index_offset + bm.index_count]
        flat_bm_slice = slice(0, n_vertex) if bm.brain_structure == CORTEX_LEFT_LABEL else slice(n_vertex, None)
        flat_data[:, flat_bm_slice][:, bm.vertex_indices._indices] = bm_data

    return flat_data


# ----------------------------------------------------------------------------# 
# -------------------          Compare CAP States          -------------------# 
# ----------------------------------------------------------------------------# 


def parallel_nulls(cap_data, n_perm, n_workers=8, n_max_per_worker=20, seed=137):
    """ """
    n_jobs = int(np.ceil(n_perm / n_max_per_worker))

    np.random.seed(seed)
    seeds = np.random.choice(np.arange(1_000_000), size=n_jobs, replace=False)
    job_sizes = [min(n_perm - i * n_max_per_worker, n_max_per_worker) for i in range(n_jobs)]
    data = [cap_data for i in range(n_jobs)]

    null_worker = lambda args: nm.nulls.alexander_bloch(data=args[0], atlas='fslr', density='32k',
                                                        n_perm=args[1], seed=args[2])
    with mp.Pool(n_workers) as pool:
        results = pool.map(null_worker, zip(data, job_sizes, seeds))
    return np.hstack(results)


def compare_caps(cap_set, compare_cap_set, n_perm = 40, metric='pearsonr', seed = 42, n_workers = 8):
    """ """

    if isinstance(cap_set, str):
        cap_set = load_dscalar_as_flat_gifti(cap_set)

    if isinstance(compare_cap_set, str):
        compare_cap_set = load_dscalar_as_flat_gifti(compare_cap_set)

    compare_shape = (len(cap_set), len(compare_cap_set))
    compare_r = np.full(compare_shape, fill_value=np.nan)
    compare_p = np.full(compare_shape, fill_value=np.nan)

    for i, cap_data in enumerate(tqdm(cap_set, desc="Comparing CAPs")):
        # rotated_nulls = nm.nulls.alexander_bloch(data=cap_data, atlas='fslr', density='32k',
        #                                               n_perm=n_perm, seed=seed)
        rotated_nulls = parallel_nulls(cap_data, n_perm, n_workers=n_workers, n_max_per_worker=20, seed=seed)

        for j, compare_cap_data in enumerate(compare_cap_set):
            r, p  = stats.compare_images(cap_data, compare_cap_data, nulls=rotated_nulls,
                                         nan_policy='omit', metric=metric)
            compare_r[i,j], compare_p[i, j] = r, p

    return compare_r, compare_p


# ----------------------------------------------------------------------------# 
# --------------------          Plot Comparisons          --------------------# 
# ----------------------------------------------------------------------------# 

from scipy.cluster.hierarchy import leaves_list, linkage

def compare_CAP_heatmap(compare_r, compare_p, cap_label, compare_label, only_max=False, figsize=(5, 4)):
    """ """
    r_squared = compare_r ** 2

    mask = (compare_p >= 0.05) | (compare_r <= 0)

    if only_max == "1-max":
        max_indices = []
        mask_indices = np.vstack(np.where(~mask)).T.tolist()

        while len(mask_indices) > 0:

            new_max_index = mask_indices[np.argmax([r_squared[i, j] for i, j in mask_indices])]
            max_indices.append(new_max_index)

            new_mask_indices = []
            for mask_index in mask_indices:
                if (new_max_index[0] != mask_index[0]) and (new_max_index[1] != mask_index[1]):
                    new_mask_indices.append(mask_index)

            mask_indices = new_mask_indices

        mask = np.ones_like(r_squared, dtype=bool)
        for i, j in max_indices:
            mask[i, j] = False

    elif not isinstance(only_max, bool) or only_max:
        row_max_r_squared = np.max(r_squared, axis=only_max * 1, keepdims=True)
        mask = mask | (r_squared < row_max_r_squared)

    filtered_r_squared = np.where(mask, np.nan, r_squared)

    # Creating row and column labels
    rows = [f"{cap_label} CAP {i+1}" for i in range(compare_p.shape[0])]
    cols = [f"{compare_label} CAP {j+1}" for j in range(compare_p.shape[1])]

    # Plotting the heatmap
    fig = plt.figure(figsize=figsize)
    # gs = sns.heatmap(filtered_r_squared, annot=False, cmap="magma", xticklabels=cols, yticklabels=rows)

    ax = sns.heatmap(filtered_r_squared, annot=np.round(filtered_r_squared, 2), fmt=".2f", cmap="viridis",
                     xticklabels=cols, yticklabels=rows, annot_kws={"size": 8, "color": "white"})
    plt.xlabel(f"{compare_label} CAP")
    plt.ylabel(f"{cap_label} CAP")
    plt.title("R² CAP State Comparisons")
    plt.show()

    return filtered_r_squared


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
