echo "Start..."
python refined_from_data.py\
	--df_path for_proposal/splited_data/s1_C_slec.parquet\
	--assignment lap\
	--verbose\
	--key_param 25\
	--output_dir for_proposal/selected400/

python gen_img_from_rfd_obj.py\
	--path for_proposal/selected400/REFINED_obj.pickle\
	--df_path for_proposal/splited_data/s1_C_slec.parquet\
	--img_format npy\
	--dtype float\
	--normalize\
	--output_dir for_proposal/selected400/