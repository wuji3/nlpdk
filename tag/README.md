# Env 
```bash
conda env create -f env.yml
conda activate nlp-tag
```

# Data Prepare
```csv
sentence1,label
I love this movie,pos
I hate this movie,neg
```

# Train
```bash
#! /bin/bash

CUDA_DEVICE_VISIBLE=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 train.py  \
  --model_name_or_path google-bert/bert-base-cased \
  --train_file /home/xxx/nlpdk/data/tag/train_debug.csv \
  --learning_rate 16e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 256 \
  --problem_type "single_label_classification" \
  --label_smoothing_factor 0.1 \
  --fp16 True \
  --max_seq_length 512 \
  --pad_to_max_length False \
  --save_steps 5000 \
  --output_dir ./train_result/ \
  --overwrite_output_dir True \
```
problem_type: single_label_classification & multi_label_classification

# Predict & Eval
```bash
python predict.py -od /home/xxx/langdk/shopping_test -f /home/xxx/dkdata/reddit_queries_all_60w+.csv -m ./train_result_shopping_related/checkpoint-255

python analysize_afer_predict.py -s /home/xxx/dkdata/shopping_related_quora.csv -p /home/xxx/langdk/shopping_test/results_predict_shopping_related_quora.csv -caseout /home/xxx/langdk/shopping_test/prediction_analysis_bycase.csv -td /home/xxx/dkdata/shopping_related_data.csv -cateout /home/xxx/langdk/shopping_test/prediction_analysis_bycategory.csv -tk 2
```
