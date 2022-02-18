# REFINED generator
Generate REFINED 2D image representations for numerical data

## Use as a library

```python
process = Refined(working_dir="output_dir", assignment='lap')
process.fit(data)
process.plot_mapping()
process.save_mapping_for_hill_climbing()
process.generate_image(data, 'Images', 'jpg')
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

Generate images from data

```shell
python pipeline.py\
--df_path data\ADNI\ADNIClassData.parquet\
--transpose\
--output_dir output\ADNI_PPL\
--dim_reduction tSNE\
--assignment LAP\
--verbose\
--key_param 30\
--n_var_filter 1600\
--img_format npy 

```



## Input data
Data must be pd.DataFrame, with the columns being the features to be assigned. Each row will become one image file. 

Note:

the best dim-reduction for LAP is probably TSNE. 
