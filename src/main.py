import argparse
import os

import constants
import utils
import clustering
import surface_mapping as sfm
import pipeline

# ------------------------------------------------------------------- #
# --------------------   Main Specific Helpers   -------------------- #
# ------------------------------------------------------------------- #


def create_save_paths(args):
    """ """
    out_path = args.out_path
    assert os.path.exists(os.path.dirname(out_path)), f"{out_path} is invalid."
    out_path = out_path.rstrip("_.")
    prefix = os.path.basename(out_path)

    isc_threshold = args.isc_threshold or -1 # Dummy isc for string writing purposes

    save_paths = {}
    save_paths["ROI_subset_dlabel"] = f"{out_path}_ISC_ROI_subset_T{isc_threshold * 100:0.0f}.dlabel.nii"
    save_paths["CAP_dscalar"] = f"{out_path}_CAP_states_K{{k}}.dscalar.nii"
    save_paths["CAP_pscalar"] = f"{out_path}_CAP_states_K{{k}}.pscalar.nii"

    save_paths["CAP_labels"] = f"{out_path}_CAP_states_K{{k}}_labels.npy"
    save_paths["pkl"] = f"{out_path}_CAP_states_K{{k}}.pkl"

    plot_dir = f"{out_path}_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    save_paths["ISC_plot"] = os.path.join(plot_dir, f"{prefix}_ISC_T{isc_threshold * 100:0.0f}.png")
    save_paths["pCAP_plot"] = os.path.join(plot_dir, f"{prefix}_pCAP_{{k}}")
    save_paths["dCAP_plot"] = os.path.join(plot_dir, f"{prefix}_dCAP_{{k}}")
    save_paths["PAC_plot"] = os.path.join(plot_dir, f"{prefix}_PAC_optimize_{{k}}.png")

    return save_paths


def get_arguments(test_set = None):
    """ """
    parser = argparse.ArgumentParser(prog='CAP_analysis',
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
    parser.add_argument('-v', "--verbose", dest='verbose', action="store", type=int, default=1,
                        required=False, help="Verbosity")
    parser.add_argument('-s', "--seed", dest='seed', action="store", type=int, default=1,
                        required=False, help="Random seed")
    parser.add_argument('-k', "--set-k", dest='set_k', action="store", type=int, default=None,
                        required=False, help="Number of clusters K to use in KMeans, default finds optimal K with CKM")

    #TODO: Remove test set options
    if test_set is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(test_set)

    # TODO: Add CKM save pkl feature
    # TODO: implement argument checks

    args.pbar = args.verbose > 0

    args.title = args.title or os.path.basename(args.out_path)

    return args


# ------------------------------------------------------------------- #
# --------------------            Main            -------------------- #
# ------------------------------------------------------------------- #


def main():
    """ """
    args = get_arguments()
    save_paths = create_save_paths(args)
    cifti_paths = pipeline.get_cifti_paths(args.ciftis)

    raw_cifti_data_array, ROI_labels = utils.load_cifti_arrays(cifti_paths, pbar=args.pbar)

    preprocess_arguments = {"crop_length": 0} #TODO: Make this into real argparse with specific args
    cifti_array = pipeline.preprocess_cifti_array(raw_cifti_data_array, pbar=args.pbar, **preprocess_arguments)

    template_cifti = nb.load(cifti_paths[0])

    if args.isc_threshold:
        ROI_subset, isc_df = pipeline.ISC_subset(cifti_array, ROI_labels, isc_threshold=args.isc_threshold, pbar=args.pbar)
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

    #TODO: Add optimal K plot (PAC optim)
    CAP_states, CAP_labels = clustering.find_CAP_states(cifti_array, ROI_labels,
                                                          set_k=args.set_k,
                                                          ROI_subset=ROI_subset,
                                                          seed=args.seed,
                                                          pbar=args.pbar,
                                                          save_plot_path=save_paths["PAC_plot"])

    k = len(CAP_states)
    with open(save_paths["CAP_labels"].format(k=k), 'wb') as f:
        np.save(f, CAP_labels.reshape(cifti_array.shape[0], -1))

    with open(save_paths["pkl"].format(k=k), 'wb') as file:
        obj = [CAP_states, CAP_labels, isc_df, ROI_labels, cifti_paths]
        pickle.dump(obj, file)

    utils.write_CAP_scalars(CAP_states, save_paths["CAP_pscalar"], cifti=template_cifti)
    # TODO: Add plotting functions for CAP states (frac occ, etc.)
    plots.create_CAP_state_plots(CAP_states, CAP_labels, ROI_labels, template_cifti,
                                     save_path=save_paths["pCAP_plot"])

    if args.dtseries:
        dtseries_paths = pipeline.get_cifti_paths(args.dtseries)
        template_dtseries = nb.load(dtseries_paths[0])
        dCAP_states = pipeline.create_dCAP_states(cifti_array, CAP_labels, dtseries_paths)
        utils.write_CAP_scalars(dCAP_states, save_paths["CAP_dscalar"], cifti=template_dtseries)
        # TODO: Add plotting functions for CAP states (frac occ, etc.)
        plots.create_CAP_state_plots(dCAP_states, CAP_labels, ROI_labels, template_dtseries,
                                         save_path=save_paths["dCAP_plot"])


if __name__ == '__main__':
    print()
    main()
