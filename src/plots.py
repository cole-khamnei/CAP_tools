import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import os
# import sys
# sys.path.append(os.path.dirname(__file__))

from . import utils
from . import surface_mapping as sfm

# ------------------------------------------------------------------- #
# --------------------       Plot Functions      -------------------- #
# ------------------------------------------------------------------- #


def ISC_plot(isc_df: pd.DataFrame, isc_threshold: float, template_cifti, save_path = None, title=""):
    """ """
    isc_values = utils.cifti_map(isc_df["roi"], isc_df["isc"], template_cifti)
    isc_thresholded_df = isc_df.query(f"isc >= {isc_threshold}")
    isc_thresholded_values = utils.cifti_map(isc_thresholded_df["roi"], isc_thresholded_df["isc"], template_cifti)
    
    fig = plt.figure(figsize=(12, 3))
    a1 = fig.add_axes([0.1, 0.15, 0.2, 0.80])
    a0 = fig.add_axes([0.2, 0.15, 0.85, 0.85])
    
    a0, p = sfm.surface_plot(isc_values, cmap=plt.cm.plasma, color_range=[0, 0.6], ax=a0)
    a0.set_title(f"{title} ISC Median\n ({len(isc_thresholded_df)} ROIs >= {isc_threshold})")
    sfm.surface_plot(isc_thresholded_values, outline=True, cbar=False, cmap=plt.cm.plasma, 
                     color_range=[0, 0.6], ax=a0, p=p, cbar_label='ISC r')
    
    sns.kdeplot(isc_df.isc, ax=a1, fill=True, color="purple")
    sns.despine(ax=a1)
    a1.set_xlabel("ISC r")
    a1.axvline(isc_threshold, linestyle=":", color="w")
    a1.set_title(f"ISC Values\nThreshold = {isc_threshold:.2f}")
    a1.set_ylim(None, 1.2 * a1.get_ylim()[1])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')


def create_CAP_state_plots(CAP_states: np.ndarray, CAP_labels: np.ndarray,
                           ROI_labels, template_cifti, save_path: str = None, title=""):
    """ """
    # TODO: Create CAP occupancy graphs (dwell time also, etc)
    # TODO: Create

    cmap = sns.color_palette("icefire", as_cmap=True)

    k = len(CAP_states)
    fig, axes = plt.subplots(k, 1, figsize=(12, k * 3.5))
    plt.subplots_adjust(hspace=-0.0)
    for i, CAP_state in enumerate(CAP_states):
        CAP_state_values = utils.cifti_map(ROI_labels, CAP_state, template_cifti)
        ax, p = sfm.surface_plot(CAP_state_values, cmap=cmap, ax=axes[i],
                                 cbar_kws=dict(pad=-0.08)) #, color_range=[-0.75, 0.75])
        ax.set_title(f"{title} CAP State {i + 1}", y=0.83)

    if save_path:
        save_path = save_path.format(k=k) if "{k}" in save_path else save_path
        fig.savefig(f"{save_path}_states.png", bbox_inches='tight')

    #TODO: Add fractional occupancy




# ------------------------------------------------------------------- #
# --------------------            END            -------------------- #
# ------------------------------------------------------------------- #
