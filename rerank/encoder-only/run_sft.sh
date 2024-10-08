CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run.py \
--model_name_or_path BAAI/bge-reranker-v2-m3 \
--train_data shoppal/embedding-index \
--sub_data_name source2.3 \
--learning_rate 6e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 8 \
--dataloader_drop_last True \
--train_group_size 4 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 500 \
--save_steps 500 \
--overwrite_output_dir True \
