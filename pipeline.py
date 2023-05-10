"""
Pipeline icluding:
    generate REFINED from data;
    generate images;
    enlarge images.

Example code:
python pipeline.py --df_path data\ADNI\ADNIClassData.parquet --transpose --output_dir output\ADNI_PPL --dim_reduction tSNE --assignment LAP --verbose --key_param 30 --n_var_filter 1600 --img_format npy 

"""

from refined.refined_CLI import pipeline
from refined.args import PipelineArgs

args = PipelineArgs().parse_args()

pipeline(args)
