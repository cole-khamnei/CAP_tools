import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

# import os
# import sys
# sys.path.append(os.path.dirname(__file__))

from . import utils
from . import surface_mapping as sfm

from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------------------------------------------------------------------# 
# --------------------           Plot Functions           --------------------# 
# ----------------------------------------------------------------------------# 


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

    try:
        for i, CAP_state in enumerate(CAP_states):
            CAP_state_values = utils.cifti_map(ROI_labels, CAP_state, template_cifti)
            ax, p = sfm.surface_plot(CAP_state_values, cmap=cmap, ax=axes[i],
                                     cbar_kws=dict(pad=-0.08)) #, color_range=[-0.75, 0.75])
            ax.set_title(f"{title} CAP State {i + 1}", y=0.83)
    except:
        return

    if save_path:
        save_path = save_path.format(k=k) if "{k}" in save_path else save_path
        fig.savefig(f"{save_path}_states.png", bbox_inches='tight')

    #TODO: Add fractional occupancy


def get_CAP_occupancy(CAP_labels):
    """ """
    num_CAPs = np.max(CAP_labels) + 1
    return np.array([np.bincount(TR, minlength=num_CAPs) / CAP_labels.shape[0]
                     for TR in CAP_labels.T])


def CAP_trajectory_plot(CAP_labels, analysis_label, n_TRs=None, save_path=None):
    """ """
    num_CAPs = np.max(CAP_labels) + 1
    CAP_occupancy = get_CAP_occupancy(CAP_labels)
    CAP_modes = scipy.stats.mode(CAP_labels, axis=0)[0]

    if n_TRs is None:
        n_TRs = len(CAP_occupancy)

    fig, (ax, a1) = plt.subplots(2, 1, figsize=(12, 2.5), height_ratios=[1, 0.1])
    fig.tight_layout(h_pad=-2)
    sns.heatmap(CAP_occupancy[:n_TRs].T * 100, ax=ax,
                cbar_kws=dict(pad=0.01, shrink=0.8, label="Occupancy Percent"))
    ax.plot(CAP_modes[:n_TRs] + 0.5, color="w", alpha=1, linewidth=0.5)
    ax.set(xticklabels=[]);
    ax.tick_params(left=False, bottom=False, labelsize=8)
    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels([f"CAP {i + 1}" for i in range(num_CAPs)], rotation=0, size=7)
    ax.set_title("CAP State Occupancy: " + analysis_label)

    pal = sns.color_palette("magma", as_cmap=True)
    colors = pal(CAP_occupancy.max(axis=1) / np.max(CAP_occupancy))[:n_TRs]
    for i, c in enumerate(colors):
        a1.bar(i, 1, color=c, width=1)
    a1.set_xlim(0, len(colors) * 1.19) # 150)
    xticks = a1.get_xticks()
    a1.set_xticks(xticks[xticks <= len(colors)])
    a1.set_xlabel("TR" + " " * 40)
    a1.set_ylabel("Occupancy %", rotation=0, va="center", ha="right", size=8)
    plt.setp(a1.spines.values(), color=None)
    # a1.axes.get_yaxis().set_visible(False)
    a1.set(yticklabels=[]);

    if save_path:
        fig.savefig(save_path)


def CAP_occupancy_distributions(CAP_labels, analysis_label, save_path=None):
    """ """
    CAP_occupancy = get_CAP_occupancy(CAP_labels)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3))
    fig.tight_layout()
    sns.kdeplot(CAP_occupancy.max(axis=1), fill=True, color="Purple", bw_method=0.2, ax=a0)
    a0.set(xlabel="Fractional Occupancy (Percent of Subjects)",
           title="Percent Subjects of Most Occupied State at Each TR")
    sns.boxplot(data=CAP_occupancy, ax=a1)
    a1.set_xticks(range(CAP_occupancy.shape[1]))
    a1.set_xticklabels(range(1, CAP_occupancy.shape[1] + 1))

    a1.set(ylabel="Fractional Occupancy", xlabel="CAP State",
           title="CAP State Fractional Occupancy Distributions")

    fig.suptitle(f"CAP State Occupancy Distributions: {analysis_label}", y=1.15)
    if save_path:
        fig.savefig(save_path)


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # Sort the array and calculate the cumulative proportion of the population
    if array.ndim > 1:
        return np.array([gini(a_i) for a_i in array])

    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1) / n
    # Calculate the area under the Lorenz curve
    area_under_lorenz = (array * index).sum() / array.sum()
    # Calculate the Gini coefficient
    return 1 - 2 * area_under_lorenz


def CAP_occupancy_timeseries(CAP_labels, analysis_label, save_path=None):
    """ """

    CAP_occupancy = get_CAP_occupancy(CAP_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5),
                             gridspec_kw={'width_ratios': [10, 1], "height_ratios": [2.5, 1]})
    fig.tight_layout(w_pad=-2)
    (a0, a1) = axes[0]
    a0.plot(np.mean(CAP_occupancy, axis=1), alpha=0.5, linestyle="--")
    a0.plot(np.median(CAP_occupancy, axis=1), color="C0", label="Median Occupancy Fraction")
    a0.fill_between(np.arange(len(CAP_occupancy)),
                     np.percentile(CAP_occupancy, 5, axis=1),
                     np.percentile(CAP_occupancy, 95, axis=1), alpha=0.2, label="State Occupancy - 90% CI")
    a0.plot(np.arange(CAP_occupancy.shape[0]), np.max(CAP_occupancy, axis=1),
            color="C3", alpha=0.4, linestyle="--", linewidth=1,
            label="Maximally Occupied State")

    chi_s, p_s = scipy.stats.chisquare(CAP_occupancy, 1/12, axis=1)
    q_s = scipy.stats.false_discovery_control(p_s)
    s = f"{np.sum(q_s < 0.05)} Significant TRs by Chi-Squared"

    a0.set(xlabel="TR", ylabel="Occupancy Fraction", xlim=(-10, CAP_occupancy.shape[0] + 5))
    a0.set_title(f"Occupancy Fraction over Time\n{s}")
    a0.legend()

    sns.kdeplot(y=CAP_occupancy.ravel(), bw_method=0.05, fill=True, ax=a1)
    sns.kdeplot(y=np.max(CAP_occupancy, axis=1).ravel(), color="C3", bw_method=0.05, fill=True, ax=a1)
    a1.set(xlabel="Occupancy Fraction", ylim=(a0.get_ylim()))
    a1.yaxis.tick_right()

    (a0, a1) = axes[1]
    fig.tight_layout(w_pad=-2)
    CAP_gini = gini(CAP_occupancy)
    a0.plot(CAP_gini)
    a0.set(xlabel="TR", ylabel="Gini Index", title="CAP Occupancy Gini Index")
    sns.kdeplot(y=CAP_gini, bw_method=0.1, fill=True, ax=a1)
    a1.set(xlabel="Gini Index", ylim=(a0.get_ylim()))
    a1.yaxis.tick_right()
    a1.yaxis.set_label_position("right")


    if save_path:
        fig.savefig(save_path)




# ----------------------------------------------------------------------------# 
# --------------------                END                 --------------------# 
# ----------------------------------------------------------------------------#
