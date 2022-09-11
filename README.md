# REFINED generator
Generate REFINED 2D image representations for numerical tabular data.

Better organized for internal usage. 

## Dependencies

TAP `pip install typed-argument-parser`

myToolbox, see https://github.com/Ribosome25/myToolbox . (You may clone this repo to your py library dir or the current working dir.)

## Use as a library

REFINED save all intermediate results and final images in a designated working dir. 

To generate a REFINED mapping, save the mapping, and generate images:

```python
from refined.refined import Refined
...
process = Refined(working_dir="output_dir", assignment='lap')
process.fit(data)
process.plot_mapping()
process.save_mapping_for_hill_climbing()
process.save_mapping_to_csv()
process.generate_image(data, 'Images', 'jpg')
```

Check refined.refined.py for more available arguments.

Recommend to save and load the REFINED mapping with JSON. 

```
rfd = Refined()
rfd.fit(data)
rfd.save_mapping_to_json()
...
rfd2 = Refined(working_dir='copied')
rfd2.load_from_json("REFINED_mapping.json")
```

## Use as CLI

Generate and save REFINED mapping from data

```shell
python refined_from_data.py\
--df_path data\ADNI\ADNIClassData.parquet\
--transpose\
--output_dir output\ADNI_CLI\
--dim_reduction tSNE\
--assignment LAP\
--verbose\
--key_param 30\
--n_var_filter 2500

```

Generated images from saved REFINED mapping and data

```shell
python gen_img_from_rfd.py\
--rfd_path output\ADNI_CLI\REFINED_mapping.json\
--df_path data\ADNI\ADNIClassData.parquet\
--transpose\
--img_format npy\
--dtype float\
--output_dir ReadJson

```

Generate images from data, and enlarge the RFD images for visualization:

```shell
python pipeline.py\
--df_path data\ADNI\ADNIClassData.parquet\
--transpose\
--output_dir output\ADNI_PPL\
--dim_reduction tSNE\
--assignment LAP\
--hw 50\
--verbose\
--key_param 30\
--n_var_filter 1600\
--img_format png 

```

Augmentation with different random seeds: change the seed and the output dir. 

The random seed will be passed to the dim-reduction method, such as MDS, tSNE, etc.. 

```shell
python refined_from_data.py\
--df_path data\ADNI\ADNIClassData.parquet\
--transpose\
--output_dir output\ADNI_CLI_Seed10\
--dim_reduction tSNE\
--assignment LAP\
--verbose\
--key_param 30\
--n_var_filter 2500
--seed 10

```



## Input data
Data must be pd.DataFrame, with the columns being the features to be assigned. Each row will become one image file. 

If in the data file, cols are sample index and rows are features, dont forget to add a transpose step `data = data.T` or `--transpose` in the args. 

Note:

the best dim-reduction for LAP is probably TSNE. 





## Recently used CLIs

```bash
python pipeline.py --df_path G:/integrated_model_own/data/ADNIClassData.parquet --transpose --output_dir output/ADNI_tsne_spr50 --dim_reduction tSNE --assignment LAP --hw 50 --verbose --key_param 30 --n_var_filter 1600 --img_format png 

python pipeline.py --df_path G:/integrated_model_own/data/ADNIClassData.parquet --transpose --output_dir output/ADNI_mds_spr2k-64 --dim_reduction mds --assignment LAP --hw 64 --verbose --key_param 30 --n_var_filter 2000 --img_format npy

python refined_from_data.py --df_path G:/integrated_model_own/data/ADNIClassData.parquet --transpose --output_dir output/ADNI_mds_spr4k-64_s0 --dim_reduction mds --assignment LAP --verbose --key_param 30 --n_var_filter 4000 --seed 0

python pipeline.py --df_path G:/integrated_model_own/data/CCLE/tmm_normalized_logcpm_final.parquet --transpose --output_dir output/CCLE_tsne_8k100_rna --dim_reduction tSNE --assignment LAP --hw 100 --verbose --key_param 30 --n_var_filter 8000 --img_format npy

python gen_img_from_rfd.py --rfd_path output/CCLE_tsne_8k100_rna/REFINED_mapping.json --df_path G:/integrated_model_own/data/CCLE/Protein_Quant_final.parquet --transpose --img_format npy --hw 100 --output_dir "./Prot"
```

