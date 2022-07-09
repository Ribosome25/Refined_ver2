"""
TAP Arguments

"""

from typing import List
from tap import Tap  # pip install typed-argument-parser


class RFDArgs(Tap):
    df_path: List  # The paths to the input data df files (or precomputed distances). Files will be concatenated. Also make precomputed ones a df please.
    transpose: bool  # If true, DF will be transposed.
    output_dir: str  # the working folder
    dim_reduction: str='MDS'  # Dimensional reduction method. MDS, ISOMAP, C-ISOMAP, tSNE, LLE, DiffMap ...
    distance_metric: str='correlation'  # distance metric, or precomputed.
    assignment: str='refined'  # refined or LAP. LAP will be solved with Scipy. 
    hw: int=None  # hw, default None
    verbose: bool
    key_param: float=5  # the key_param for dimreduction. N of nearest neighbors for most methods, or perplexity for tSNE. 
    n_var_filter: int=None  # max variance feature selection. how many feature with the most variance will be kept. 
    seed: int=None  # Random seed for dim reduction

class GenImgArgs(Tap):
    rfd_path: str  # the path to the REFINED object, or json
    df_path: List  # the path(s) to the data to be genearated
    transpose: bool  # If true, DF will be transposed.
    img_format: str="npy"  # image format. npy means save as np arrays. 
    hw: int=None  # hw, default None
    normalize: bool=False  # normalize each feature to 0 - 255. In the case that some features are always high/or low, or features have diff units. Usually kept True.
    zscore: bool=False  # standardize each feature to N(0, 1).
    zscore_cutoff: float=5  # zscores beyond +- cutoff will be clipped. In the case there is an extreme value thus other values all become bright/dark.
    output_dir: str='RFD_Images'  # where to store the images.
    fill_blank: str="zeros"  # fill blank pixels with zeros or mean?


# class PipelineArgs(RFDArgs, GenImgArgs):
class PipelineArgs(Tap):
    df_path: List  # The paths to the input data df files (or precomputed distances). Files will be concatenated. Also make precomputed ones a df please.
    transpose: bool  # If true, DF will be transposed.
    output_dir: str  # the working folder
    dim_reduction: str='MDS'  # Dimensional reduction method. MDS, ISOMAP, C-ISOMAP, tSNE, LLE, DiffMap ...
    distance_metric: str='correlation'  # distance metric, or precomputed.
    assignment: str='refined'  # refined or LAP. LAP will be solved with Scipy. 
    hw: int=None  # hw, default None
    verbose: bool
    key_param: float=5  # the key_param for dimreduction. N of nearest neighbors for most methods, or perplexity for tSNE. 
    n_var_filter: int=None  # max variance feature selection. how many feature with the most variance will be kept. 
    seed: int=None  # Random seed for REFINED mapping generation -> dim reduction

    img_format: str="npy"  # image format. npy means save as np arrays. 
    normalize: bool=False  # normalize each feature to 0 - 255. In the case that some features are always high/or low, or features have diff units. Usually kept True.
    zscore: bool=False  # standardize each feature to N(0, 1).
    zscore_cutoff: float=5  # zscores beyond +- cutoff will be clipped. In the case there is an extreme value thus other values all become bright/dark.
    fill_blank: str="zeros"  # fill blank pixels with zeros or mean?
