"""
Load data and genearate df. 

"""

from refined.refined import gen_mapping
from refined.args import RFDArgs

args = RFDArgs().parse_args()
gen_mapping(args)

