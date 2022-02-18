"""
Load data and genearate refined results. 

Example code:
python refined_from_data.py --df_path data\ADNI\ADNIClassData.parquet --transpose --output_dir output\ADNI_CLI --dim_reduction tSNE --assignment LAP --verbose --key_param 30 --n_var_filter 2500
"""

from refined.refined_CLI import gen_mapping_from_data
from refined.args import RFDArgs

args = RFDArgs().parse_args()
gen_mapping_from_data(args)

