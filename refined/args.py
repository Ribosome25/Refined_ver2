"""
TAP Arguments

"""

from typing import List
from tap import Tap  # pip install typed-argument-parser


class DimReduArgs(Tap):
    df_path: List  # The paths to the input data df files (or precomputed distances). Files will be concatenated. Also make precomputed ones a df please.
    dim_reduction: str='MDS'  # Dimensional reduction method
    distance_metric: str='correlation'  # distance metric, or precomputed.
    assign: str='refined'  # refined or LAP. LAP will be solved with Scipy. 
    verbose: bool
    output_dir: str  # where to save the generated files


class GenImgArgs(Tap):
    mapping_path: str  # the path to the mapping or REFINED object
    img_format: str="npy"  # image format. npy means save as np arrays. 
    dtype: str='int'  # the dtype of images. if float, the img_format must be npy
    normalize: bool  # normalize each feature to 0 - 255. In the case that some features are always high/or low, or features have diff units. Usually kept True.
    zscore: bool  # standardize each feature to N(0, 1).
    zscore_cutoff: float=5  # zscores beyond +- cutoff will be clipped. In the case there is an extreme value thus other values all become bright/dark.
    output_dir: str  # where to store the images


class PipelineArgs(DimReduArgs, GenImgArgs):
    pass
