import glob
import os
import pathlib

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import nibabel as nb
import surfplot as sfp


import constants

# ------------------------------------------------------------------- #
# --------------------          Constants        -------------------- #
# ------------------------------------------------------------------- #


SURFACE_PATHS = glob.glob(os.path.join(constants.SURFACES_DIR_PATH, "ec_con*.L.*"))
SURFACE_PATHS = {spath.split(".")[2]: (spath, spath.replace(".L.", ".R.")) for spath in SURFACE_PATHS}

SURFACES = {label: (pathlib.PosixPath(spath[0]), pathlib.PosixPath(spath[1]))
            for label, spath in SURFACE_PATHS.items()}

ROW_VIEWS = dict(layout='row', zoom=1.2, size=(4 * 275, 275), brightness=0.6, mirror_views=False)
GRID_VIEWS = dict(layout='grid', zoom=1.5, brightness=0.5)

CIFTI_NVERTEX = 32492

# ------------------------------------------------------------------- #
# --------------------    Writing Ciftis         -------------------- #
# ------------------------------------------------------------------- #


def create_cifti_cortex_axis(nvertex=CIFTI_NVERTEX):
    """ """
    bm_l = nb.cifti2.BrainModelAxis.from_surface(np.arange(nvertex), nvertex, name='cortex_left')
    bm_r = nb.cifti2.BrainModelAxis.from_surface(np.arange(nvertex), nvertex, name='cortex_right')
    bm_cortex = bm_l + bm_r

    return bm_cortex


def values_to_dscalar(values, scalar_name: str = "variable") -> nb.cifti2.cifti2.Cifti2Image:
    """ """
    assert CIFTI_NVERTEX == len(values["left"])
    assert CIFTI_NVERTEX == len(values["right"])

    bm_cortex = create_cifti_cortex_axis()
    axes = (nb.cifti2.ScalarAxis([scalar_name]), bm_cortex,)
    header = nb.cifti2.Cifti2Header.from_axes(axes)
    data = np.hstack([values["left"], values["right"]]).reshape(1, -1)

    return nb.Cifti2Image(data, header=header)


def write_values_to_dscalar(values, path, scalar_name: str = "variable"):
    """ """
    assert path.endswith("dscalar.nii")
    cifti = values_to_dscalar(values, scalar_name=scalar_name)
    cifti.to_filename(path)


def create_dlabel_map(labels, null_label="???"):
    """ """
    label_set = np.unique(np.vstack([labels["left"], labels["right"]]))
    lb_sorter = np.array([label.lstrip("LR_") for label in label_set])
    label_set_idx = np.argsort(lb_sorter)
    
    int_to_label_map = dict(enumerate(label_set[label_set_idx]))
    label_to_int_map = {v: k for k,v in int_to_label_map.items()}

    return label_to_int_map


def create_cifti_label_map(label_to_int_map: dict, cmap = None):
    """ """
    if cmap is None:
        cmap = plt.cm.viridis.resampled(len(label_to_int_map))

    return {v: (k, (*cmap.colors[i],)) for i, (k, v) in enumerate(label_to_int_map.items())}


def labels_to_dlabel(labels, label_to_int_map: dict = None, cmap=None, label_name="label"):
    """ """
    assert CIFTI_NVERTEX == len(labels["left"])
    assert CIFTI_NVERTEX == len(labels["right"])
    
    if label_to_int_map is None:
        label_to_int_map = create_dlabel_map(labels)

    cifti_label_map = create_cifti_label_map(label_to_int_map, cmap=cmap)
    label_axis = nb.cifti2.LabelAxis((label_name,), (cifti_label_map,))
    bm_cortex = create_cifti_cortex_axis()
    header = nb.cifti2.Cifti2Header.from_axes((label_axis, bm_cortex))
    label_ints = np.array([[*map(label_to_int_map.get, labels[k])] for k in ["left", "right"]])

    return nb.Cifti2Image(label_ints.reshape(1, -1), header=header)


def write_labels_to_dlabel(labels, path, label_name: str = "label", cmap=None):
    """ """
    assert path.endswith("dlabel.nii")
    cifti = labels_to_dlabel(labels, label_name=label_name, cmap=cmap)
    cifti.to_filename(path)


def series_to_dtseries(values, start=0, step=1):
    """ """
    assert CIFTI_NVERTEX == len(values["left"])
    assert CIFTI_NVERTEX == len(values["right"])

    size = values["left"].shape[-1]
    assert values["right"].shape[-1] == size
    
    bm_cortex = create_cifti_cortex_axis()
    series = nb.cifti2.SeriesAxis(start=start, step=step, size=size)
    
    header = nb.cifti2.Cifti2Header.from_axes((series, bm_cortex))
    data = np.vstack([values["left"], values["right"]]).T

    return nb.Cifti2Image(data, header=header)


def write_series_to_dtseries(series, path, start=0, step=1):
    """ """
    assert path.endswith("dtseries.nii")
    cifti = series_to_dtseries(series, start=start, step=step)
    cifti.to_filename(path)


# ------------------------------------------------------------------- #
# --------------------    Parcellation Schemes   -------------------- #
# ------------------------------------------------------------------- #

# https://balsa.wustl.edu/about/fileTypes#:~:text=GIFTI%20is%20an%20established%20data,label%20data%20for%20each%20vertex.
# https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb

def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def get_cifti_values(cifti):
    """ """
    data = cifti.get_fdata()
    axis = cifti.header.get_axis(1)
    lh_values = surf_data_from_cifti(data, axis, 'CIFTI_STRUCTURE_CORTEX_LEFT').reshape(-1)
    rh_values = surf_data_from_cifti(data, axis, 'CIFTI_STRUCTURE_CORTEX_RIGHT').reshape(-1)
    return {"left": lh_values, "right": rh_values}


def load_parcellation(path: str):
    """ """
    assert path.endswith("dlabel.nii")
    cifti = nb.load(path)
    data = cifti.get_fdata()
    axes = [cifti.header.get_axis(i) for i in range(cifti.ndim)]
    left = surf_data_from_cifti(data, axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT').reshape(-1)
    right = surf_data_from_cifti(data, axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT').reshape(-1)

    cifti_map = dict(next(cifti.header.matrix[0].named_maps).label_table)
    # cifti_label_map = {k: v.label for k, v in cifti_map.items()}
    cifti_color_map = {k: v.rgba for k, v in cifti_map.items()}

    cifti_label_map = {}
    for k, v in cifti_map.items():
        label = v.label
        if "LEFT" in label or "RIGHT" in label:
            hemistr = "L_" if "LEFT" in label else "R_"
            tokens = label.split("_")[::-1][1:]
            label = hemistr + "".join([t.title() for t in tokens])

        cifti_label_map[k] = label

    return left, right, cifti_label_map


# ------------------------------------------------------------------- #
# --------------------      Surface Plotting     -------------------- #
# ------------------------------------------------------------------- #


def handle_nans(values, eps=np.finfo(float).eps):
    """ """
    L_ni = np.isnan(values["left"])
    R_ni = np.isnan(values["right"])

    if not any(L_ni | R_ni):
        return values

    new_values = {"left": np.zeros(len(L_ni)), "right": np.zeros(len(R_ni))}
    new_values["left"][~L_ni] = values["left"][~L_ni] + eps
    new_values["right"][~R_ni] = values["right"][~R_ni] + eps
    return new_values


def surface_plot(values, ax = None, cmap = plt.cm.coolwarm, cbar=True, outline=False,
                   surface="very_inflated", views=ROW_VIEWS, color_range=None,
                   uniform_cbar=True, p=None, cbar_label=None, cbar_kws={}, **kwargs):
    """ """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    if isinstance(values["left"][0], np.str_):
        labels = sorted(set(values["left"]).union(set(values["right"])))


        label_to_int_map = create_dlabel_map(values)
        new_values = {}
        new_values["left"] = np.array([label_to_int_map[l] for l in values["left"]])
        new_values["right"] = np.array([label_to_int_map[l] for l in values["right"]])
        values = new_values

    if "left" not in values or len(values["left"]) == 0:
        values["left"] = np.full(shape=CIFTI_NVERTEX, fill_value=np.nan)

    if "right" not in values or len(values["right"]) == 0:
        values["right"] = np.full(shape=CIFTI_NVERTEX, fill_value=np.nan)

    if p is None:
        p = sfp.Plot(*SURFACES[surface], **views)

    # TODO: CBAR label not working
    p.add_layer(handle_nans(values), cmap=cmap, cbar=cbar, color_range=color_range, cbar_label=cbar_label, **kwargs)

    if outline:
        outlines = {"left": (values["left"] == 0) * 1, "right": (values["right"] == 0) * 1}
        p.add_layer(values, cmap=plt.cm.grey, as_outline=True, cbar=False)

    add_surface_to_ax(ax, p, cbar=cbar)
    if cbar:
        surface_add_cbar(values, ax, cmap=cmap, color_range=color_range, uniform_cbar=uniform_cbar, **cbar_kws)
    return ax, p


def surface_add_cbar(values, ax, cmap, color_range=None, location="bottom",
                     uniform_cbar=True, decimals=2, shrink=0.3, aspect=20, pad=0.05,
                     fraction=0.05, n_ticks=3, **cbar_kws):
    """ """
    if color_range is None:
        vmin = min(np.nanmin(values["left"]), np.nanmin(values["right"]))
        vmax = max(np.nanmax(values["left"]), np.nanmax(values["right"]))

        if uniform_cbar and vmin < 0:
            vmax = max(abs(vmin), vmax)
            vmin = -vmax

        vmin = np.round(vmin, decimals)
        vmax = np.round(vmax, decimals)

        color_range = [vmin, vmax]

    norm = mpl.colors.Normalize(*color_range)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ticks = np.linspace(*color_range, n_ticks)
    cb = plt.colorbar(sm, ax=ax, ticks=ticks, location=location, pad=pad,
                      fraction=fraction, shrink=shrink, aspect=aspect, **cbar_kws)


def add_surface_to_ax(ax, p: sfp.Plot, cbar=True):
    """ """
    pz = p.render()
    pz._check_offscreen()
    x = pz.to_numpy(transparent_bg=True, scale=(2, 2))
    ax.imshow(x)
    ax.axis("off")


# ------------------------------------------------------------------- #
# --------------------            End            -------------------- #
# ------------------------------------------------------------------- #
