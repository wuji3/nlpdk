#! /bin/bash

python -m finetune.eval \
--encoder /home/duke/nlpdk/embedding/embedding-sft-output/2024-07-19-20-01-25 \
--fp16 \
--k 100 \
--corpus_data shoppal/embedding-index \
--corpus_data_name source2.3-corpus \
--query_data shoppal/embedding-index \
--query_data_name source2.3 \
--max_query_length 512 \
--max_passage_length 512 \
--split test \
--batch_size 6400 \
--roc_save_path '' 