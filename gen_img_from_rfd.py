"""
Read the pickle refiend obj and generate images.

Example code:
python gen_img_from_rfd.py --rfd_path output\ADNI\REFINED_mapping.json --df_path data\ADNI\ADNIClassData.parquet --transpose --img_format png  --output_dir ReadJson
python gen_img_from_rfd.py --rfd_path output\ADNI_CLI\REFINED_mapping.json --df_path data\ADNI\ADNIClassData.parquet --transpose --img_format npy --dtype float --output_dir ReadJson

"""

from refined.refined_CLI import gen_images_from_json
from refined.args import GenImgArgs

args = GenImgArgs().parse_args()

gen_images_from_json(args)
