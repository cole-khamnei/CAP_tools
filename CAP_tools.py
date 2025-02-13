import argparse
import os
import glob

import nibabel as nb
import numpy as np
import dill as pickle

from src import constants
from src import utils
from src import plots
from src import pipeline
from src import surface_mapping as sfm


# \section argument defaults

DEFAULT_VERBSOITY = 1
DEFAULT_MIN_OPT_K = 7
DEFAULT_K_MAX = 15
DEFAULT_N_REPS = 40
DEFAULT_SEED = 1

# ----------------------------------------------------------------------------# 
# ----------------           Main Specific Helpers            ----------------# 
# ----------------------------------------------------------------------------# 


def create_save_paths(args):
    """ """
    out_path = args.out_path
    assert os.path.exists(os.path.dirname(out_path)), f"{out_path} is invalid."
    out_path = out_path.rstrip("_.")
    prefix = os.path.basename(out_path)

    isc_threshold = args.isc_threshold or -1 # Dummy isc for string writing purposes

    dist_prefix = args.distance_metric[:3]

    save_paths = {}
    save_paths["ROI_subset_dlabel"] = f"{out_path}_ISC_ROI_subset_T{isc_threshold * 100:0.0f}.dlabel.nii"
    save_paths["CAP_dscalar"] = f"{out_path}_CAP_states_K{{k}}_{dist_prefix}.dscalar.nii"
    save_paths["CAP_pscalar"] = f"{out_path}_CAP_states_K{{k}}_{dist_prefix}.pscalar.nii"

    save_paths["CAP_labels"] = f"{out_path}_CAP_states_K{{k}}_{dist_prefix}_labels.npy"
    save_paths["pkl"] = f"{out_path}_CAP_states_K{{k}}_{dist_prefix}.pkl"
    save_paths["params"] = f"{out_path}_CAP_states_{dist_prefix}.params"

    plot_dir = f"{out_path}_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    save_paths["ISC_plot"] = os.path.join(plot_dir, f"{prefix}_ISC_T{isc_threshold * 100:0.0f}.png")
    save_paths["pCAP_plot"] = os.path.join(plot_dir, f"{prefix}_pCAP_{{k}}_{dist_prefix}")
    save_paths["dCAP_plot"] = os.path.join(plot_dir, f"{prefix}_dCAP_{{k}}_{dist_prefix}")
    save_paths["PAC_plot"] = os.path.join(plot_dir, f"{prefix}_PAC_optimize_{{k}}_{dist_prefix}.png")

    return save_paths


def save_params(save_paths, args):
    """ """
    with open(save_paths["params"], "w") as param_file:
        param_file.write(f"title :: {args.title}\n")
        for k, v in vars(args).items():
            if k in ["ciftis", "dtseries", "title"]:
                continue
            param_file.write(f"{k} :: {v}\n")

        for path_set, path_set_label in zip([args.ciftis, args.dtseries],
                                            ["cluster ciftis:", "dtseries"]):

            param_file.write(f"\n{path_set_label}:\n")
            if path_set[0].endswith(".txt"):
                param_file.write(f"   txt file path: {path_set[0]}\n")

            for path in pipeline.get_cifti_paths(path_set):
                param_file.write(f"   {path}\n")


def check_outputs_exist(save_paths, k="*"):
    """ """
    outputs_exist = False
    for dtype in ["CAP_pscalar", "pkl", "CAP_labels", "params"]:
        path = save_paths[dtype].format(k=k)

        path_exists = len(glob.glob(path)) > 0
        if path_exists:
            f"{dtype} path already exists: {path}. To overwrite use '--overwrite' flag"

        outputs_exist = path_exists or outputs_exist

    return outputs_exist


def get_arguments(test_set: list = None):
    """

    test_set: easy arg for testing get_arguments function and whole pipeline
    """
    parser = argparse.ArgumentParser(prog='CAP-tools',
                                     description='Identifies CAP states for a given set of CIFTIs')
    parser.add_argument('-c', "--ciftis", dest='ciftis', action="extend", nargs="+", type=str, required=True,
                        help="Txt file with paths of cifti files or cifti glob path")
    parser.add_argument('-o', "--out", dest='out_path', action="store", type=str, required=True,
                        help="Output file prefix e.g. 'path/to/dir/file_prefix'")
    parser.add_argument('-r', "--roi-subset", dest='ROI_subset_path', action="store", type=str,
                        required=False, help="ROI Subset dlabel path.")
    parser.add_argument('-i', "--isc-threshold", dest="isc_threshold", action="store", nargs='?',
                        const=0.15, default=None, required=False, type=float,
                        help="Subset ROIs using an ISC threshold.")
    parser.add_argument('-d', "--dtseries", dest='dtseries', action="extend", nargs="+", type=str, required=False,
                        help="Txt file with paths of dtseries files or glob path")
    parser.add_argument('-t', "--title", dest='title', action="store", type=str, default=None,
                        required=False, help="Title for plots")
    parser.add_argument('-v', "--verbose", dest='verbose', action="store", type=int, default=DEFAULT_VERBSOITY,
                        required=False, help="Verbosity")
    parser.add_argument('-s', "--seed", dest='seed', action="store", type=int, default=DEFAULT_SEED,
                        required=False, help="Random seed")
    parser.add_argument("--n-reps", dest='n_reps', action="store", type=int, default=DEFAULT_N_REPS,
                        required=False, help="Number of K-means repitions per K")
    parser.add_argument("--k-max", dest='kmax', action="store", type=int, default=DEFAULT_K_MAX,
                        required=False, help="Max K to check.")
    parser.add_argument("--k-min-opt", dest='min_opt_k', action="store", type=int, default=DEFAULT_MIN_OPT_K,
                        required=False, help="min K to check.")
    parser.add_argument('-k', "--set-k", dest='set_k', action="store", type=int, default=None,
                        required=False, help="Number of clusters K to use in KMeans, default finds optimal K with CKM")
    parser.add_argument("--dry-run", dest='dry_run', action="store_true", default=False,
                        required=False, help="Runs a dry of the program, checking paths but not doing any anaysis.")
    parser.add_argument("--overwrite", dest='overwrite', action="store_true", default=False,
                        required=False, help="Over writes outputs.")
    parser.add_argument('--dist', dest='distance_metric', action="store", type=str, required=False,
                        default="cosine", help="KMeans distance metric.")

    #TODO: Identify problems with GLEW library or find way to check (causes seg faults though :/  )
    parser.add_argument("--no-plots", dest='no_plots', action="store_true", default=False,
                        required=False, help="Specifies to skip plotting in case VTK/GLEW lib is messed up.")

    if test_set is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(test_set)

    # TODO: implement argument checks
    args.distance_metric = args.distance_metric.lower().strip()
    assert args.distance_metric in constants.VALID_DISTANCE_METRICS

    assert args.min_opt_k < args.kmax

    args.pbar = args.verbose > 0
    args.title = args.title or os.path.basename(args.out_path)

    return args


# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


def main():
    """ """
    args = get_arguments()
    save_paths = create_save_paths(args)
    cifti_paths = pipeline.get_cifti_paths(args.ciftis)

    cifti_paths = utils.cache_tmp_path(cifti_paths)

    if args.dry_run:
        print("Dry run complete. Valid arguments given.")
        return

    if not args.overwrite and check_outputs_exist(save_paths):
        print("Ouputs already exists. Terminating CAP-tools.")
        return

    raw_cifti_data_array, ROI_labels = utils.load_cifti_arrays(cifti_paths, pbar=args.pbar)

    preprocess_arguments = {"crop_length": 0} #TODO: Make this into real argparse with specific args
    cifti_array = pipeline.preprocess_cifti_array(raw_cifti_data_array, pbar=args.pbar, **preprocess_arguments)

    template_cifti = nb.load(cifti_paths[0])

    if args.isc_threshold:
        ROI_subset, isc_df = pipeline.ISC_subset(cifti_array, ROI_labels, isc_threshold=args.isc_threshold, pbar=args.pbar)

        if not args.no_plots:
            plots.ISC_plot(isc_df, args.isc_threshold, template_cifti,
                           save_path=save_paths["ISC_plot"], title=args.title)

        ROI_subset_values = utils.cifti_map(ROI_subset, ROI_subset, template_cifti, fill_value="???")
        # TODO: migrate away from SFM???
        sfm.write_labels_to_dlabel(ROI_subset_values, save_paths["ROI_subset_dlabel"],
                                   label_name=f"ISC_subset_{args.isc_threshold}")
        # TODO: create and output ISC dscalar
    elif args.ROI_subset_path:
        # TODO: Load in ROI subset from given dlabel
        raise NotImplementedError
    else:
        ROI_subset = None
        isc_df = []

    from src import clustering
    CAP_states, CAP_labels = clustering.find_CAP_states(cifti_array, ROI_labels,
                                                          set_k=args.set_k,
                                                          ROI_subset=ROI_subset,
                                                          seed=args.seed,
                                                          pbar=args.pbar,
                                                          n_reps=args.n_reps,
                                                          kmax=args.kmax,
                                                          min_opt_k=args.min_opt_k,
                                                          save_plot_path=save_paths["PAC_plot"])

    k = len(CAP_states)
    with open(save_paths["CAP_labels"].format(k=k), 'wb') as f:
        np.save(f, CAP_labels.reshape(cifti_array.shape[0], -1))

    with open(save_paths["pkl"].format(k=k), 'wb') as file:
        obj = [CAP_states, CAP_labels, isc_df, ROI_labels, cifti_paths]
        pickle.dump(obj, file)

    utils.write_CAP_scalars(CAP_states, save_paths["CAP_pscalar"], cifti=template_cifti)
    # TODO: Add plotting functions for CAP states (frac occ, etc.)
    if not args.no_plots:
        plots.create_CAP_state_plots(CAP_states, CAP_labels, ROI_labels, template_cifti,
                                         save_path=save_paths["pCAP_plot"])

    if args.dtseries:
        dtseries_paths = pipeline.get_cifti_paths(args.dtseries)
        dtseries_paths = utils.cache_tmp_path(dtseries_paths, write_cache=True)
        template_dtseries = nb.load(dtseries_paths[0])
        dCAP_states = pipeline.create_dCAP_states(cifti_array, CAP_labels, dtseries_paths)
        utils.write_CAP_scalars(dCAP_states, save_paths["CAP_dscalar"], cifti=template_dtseries)
        # TODO: Add plotting functions for CAP states (frac occ, etc.)

        if not args.no_plots:
            plots.create_CAP_state_plots(dCAP_states, CAP_labels, ROI_labels, template_dtseries,
                                             save_path=save_paths["dCAP_plot"])

        # TODO: Fix GLEW initialize error: most likely make plot CAP function
        # that works on outputs of CAPs

    save_params(save_paths, args)
    # IF .params file does not exist, then program did not finish correctly.


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
