echo "Start..."
# python refined_from_data.py\
	# --df_path for_proposal/splited_data/s1_C_test.parquet\
	# --assignment lap\
	# --verbose\
	# --key_param 25\
	# --output_dir for_proposal/

python gen_img_from_rfd_obj.py\
	--path for_proposal/REFINED_obj.pickle\
	--df_path for_proposal/splited_data/s1_C_test.parquet\
	--img_format png\
	--normalize\
	--output_dir for_proposal/