"""


"""
import os
from os.path import join as pj
import pickle
from refined.args import RFDArgs, GenImgArgs, PipelineArgs
from refined.refined import Refined
from refined.feature_selection import var_filter
from refined.write_videos import enlarge_images
from myToolbox.Io import check_path_exists, read_df_list, float_to_int, parent_dir

import cmapy

#%%
def gen_mapping_from_data(args: RFDArgs):
    check_path_exists(args.output_dir)
    rfd = Refined(
        dim_reduction=args.dim_reduction,
        distance_metric=args.distance_metric,
        assignment=args.assignment,
        hw=args.hw,
        working_dir=args.output_dir,
        verbose=args.verbose,
        seed=args.seed
        )

    data = read_df_list(args.df_path)
    if args.transpose:
        data = data.T
    if args.n_var_filter is not None:
        data = var_filter(data)

    key_param = float_to_int(args.key_param)
    rfd.fit(data, key_param=key_param)

    rfd.save_mapping_to_json()

    # with open(os.path.join(odir, "REFINED_obj.pickle"), 'wb') as f:
    #     pickle.dump(rfd, f)
    # with open(os.path.join(odir, "REFINED_mapping.json"), 'w') as f:
    #     json.dump(rfd.mapping_dict, f)
    return rfd


from pydoc import locate
def gen_images_from_json(args: GenImgArgs):
    if args.rfd_path.endswith(".pickle"):
        with open(args.rfd_path, 'rb') as f:
            rfd = pickle.load(f)
    elif args.rfd_path.endswith('json'):
        rfd = Refined(working_dir=parent_dir(args.rfd_path))
        rfd.load_from_json(args.rfd_path)

    data = read_df_list(args.df_path)
    if args.transpose:
        data = data.T
    dtype = locate(args.dtype)
    rfd.generate_image(
        data,
        output_folder=args.output_dir,
        img_format=args.img_format,
        dtype=dtype,
        normalize_feature=args.normalize,
        zscore=args.zscore,
        zscore_cutoff=args.zscore_cutoff,
        random_map=False, white_noise=False)

def pipeline(args: PipelineArgs):
    check_path_exists(args.output_dir)
    rfd = Refined(
        dim_reduction=args.dim_reduction,
        distance_metric=args.distance_metric,
        assignment=args.assignment,
        working_dir=args.output_dir,
        verbose=args.verbose
        )

    data = read_df_list(args.df_path)
    if args.transpose:
        data = data.T
    if args.n_var_filter is not None:
        data = var_filter(data)

    key_param = float_to_int(args.key_param)
    rfd.fit(data, key_param=key_param)

    rfd.save_mapping_to_json()
    rfd.plot_mapping()
    rfd.save_mapping_to_csv()

    dtype = locate(args.dtype)
    rfd.generate_image(
        data,
        output_folder="RFD_Images",
        img_format=args.img_format,
        dtype=dtype,
        normalize_feature=args.normalize,
        zscore=args.zscore,
        zscore_cutoff=args.zscore_cutoff,
        random_map=False, white_noise=False)

    f_list = os.listdir(pj(args.output_dir, "RFD_Images"))
    f_list = [pj(args.output_dir, "RFD_Images", x) for x in f_list if x.endswith(args.img_format)]
    enlarge_images(f_list, text_groups=data.index.tolist(), cm=cmapy.cmap('Blues_r'))