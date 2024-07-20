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