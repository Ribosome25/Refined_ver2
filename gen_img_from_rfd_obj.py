"""
Read the pickle refiend obj and generate images.
"""

from refined.refined import gen_images
from refined.args import GenImgArgs

args = GenImgArgs().parse_args()

gen_images(args)
