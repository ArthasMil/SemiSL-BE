DATA_DIR= PATH_TO_THE_TEST_FOLDER
python eval_be_single.py \
  --dataset_name WHU-Area \
  --root_dir $DATA_DIR \
  --out_dir ./test_result_temp \
  --ckpt_path PATH_TO_THE_WEIGHT_FILE

