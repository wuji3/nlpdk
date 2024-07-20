## Env Prepare
```shell
conda create -n nlp-emb python=3.10
conda activate nlp-emb

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c pytorch faiss-gpu=1.8.0 -y
conda install chardet=4.0.0 -y
pip install FlagEmbedding==1.2.9
```

## Data Prepare
1. Data For Training
    ```json
    {
        "query": "Do wireless headsets have static noise?", 
        "pos": ["If you have a wireless headset, it's possible that static is caused by interference from nearby electronics. Walk around your home or office and listen to the headset to see if your static disappears. Sometimes a problem with software is what's causing static.", "..."],
        "neg": ["Downloaded is None in fetch_url(https:\/\/www.simplyheadsets.com.au\/blog\/how-to-fix-static-coming-from-your-headset#:~:text=If%20you%20have%20a%20wireless,software%20is%20what's%20causing%20static.) function", "..."]
    }
    ```
2. Corpus For Eval
    ```json
    {
        "content": ["xxx", "xxx", "..."]
    }
    ```

## Training
### Local Data
1. If a jsonl file

    ```bash
    #! /bin/bash

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    torchrun --nproc_per_node 8 \
    -m finetune.run \
    --output_dir ./embedding-sft-output \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --train_data  /home/duke/nlpdk/data/embedding/finetune/train_toy.jsonl \
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
    ```
2. If some json files in xxx directory

    ```bash
    #! /bin/bash

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    torchrun --nproc_per_node 8 \
    -m finetune.run \
    --output_dir ./embedding-sft-output \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --train_data  /home/duke/nlpdk/data/embedding/finetune/traindata \
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
    ```
### Huggingface Data
```bash
#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 \
-m finetune.run \
--output_dir ./embedding-sft-output \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data xxx/embedding-index \
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
```

## Eval
1. Huggingface Data
    ```bash
    #! /bin/bash

    python -m finetune.eval \
    --encoder /home/duke/nlpdk/embedding/embedding-sft-output/2024-07-19-20-01-25 \
    --fp16 \
    --k 100 \
    --corpus_data xxx/embedding-index \
    --corpus_data_name source2.3-corpus \
    --query_data xxx/embedding-index \
    --query_data_name source2.3 \
    --max_query_length 512 \
    --max_passage_length 512 \
    --split test \
    --batch_size 6400 \
    --roc_save_path '' 
    ```
2. Local Data

    ```bash
    #! /bin/bash

    python -m finetune.eval \
    --encoder /home/xxx/nlpdk/embedding/embedding-sft-output/2024-07-19-20-01-25 \
    --fp16 \
    --k 100 \
    --corpus_data /home/duke/nlpdk/data/embedding/finetune/train_toy.jsonl \
    --query_data /home/duke/nlpdk/data/embedding/finetune/corpus_toy.jsonl \
    --max_query_length 512 \
    --max_passage_length 512 \
    --split test \
    --batch_size 6400 \
    --roc_save_path '' 
    ```

## Infer
```python
python finetune/predict.py
```