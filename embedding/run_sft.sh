#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 \
-m finetune.run \
--output_dir ./embedding-sft-output \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data shoppal/embedding-index \
--sub_data_name source2.3 \
--learning_rate 1e-5 \
--fp16 True \
--num_train_epochs 10 \
--per_device_train_batch_size 16 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 5 \
--negatives_cross_device True \
--use_inbatch_neg True \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 
