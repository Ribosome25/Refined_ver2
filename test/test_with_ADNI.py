"""
Run from root. 
Move it out before running. 

7/2/2022
"""

import pandas as pd
from refined.io import check_path_exists
from refined.refined import Refined

if __name__ == "__main__":
    sample = pd.read_csv("test/ADNI_test_data.csv", index_col=0).T
    sample = sample.loc[:, ~sample.columns.duplicated()]
    rfd_dir = "./test/test_ADNI_not_compact/"
    check_path_exists(rfd_dir)
    rfd = Refined(
        dim_reduction='MDS', 
        distance_metric='correlation',
        assignment='lap', 
        hw=20,
        working_dir=rfd_dir,
        verbose=True, 
        seed=0)
    rfd.fit(sample, key_param=20)  # fit refined
    rfd.plot_mapping(output_dir=rfd_dir)  # genes mapping
    rfd.save_mapping_to_csv()
    rfd.save_mapping_to_json()
    print("Image H&W", rfd.hw)

    rfd2_dir = "./test/test_ADNI_not_compact_load_json/"
    rfd2 = Refined(
        dim_reduction='MDS', 
        distance_metric='correlation',
        assignment='lap', 
        hw=None,
        working_dir=rfd2_dir,
        verbose=True, 
        seed=0)
    rfd2.load_from_json(rfd_dir + "REFINED_mapping.json")
    x0 = rfd2.generate_array(sample.iloc[0])
    x1 = rfd2.generate_array(sample.iloc[1])

    rfd2.plot_mapping(output_dir=rfd2_dir)
    rfd2.save_mapping_to_json()
    rfd2.generate_image(sample, "images", "npy", normalize_feature=True)
    rfd2.generate_image(sample, "png_imgs", "png", normalize_feature=True)
