CUDA_DEVICE_VISIBLE=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 train.py  \
  --model_name_or_path google-bert/bert-base-cased \
  --train_file /home/duke/nlpdk/data/tag/item2cate_train.csv \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.02 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 256 \
  --problem_type single_label_classification \
  --label_smoothing_factor 0.05 \
  --fp16 True \
  --max_seq_length 512 \
  --pad_to_max_length False \
  --save_steps 5000 \
  --overwrite_output_dir True \

