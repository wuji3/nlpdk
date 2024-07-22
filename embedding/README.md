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

## OHEM
```bash
python -m finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- `input_file`: json data for finetuning. This script will retrieve top-k documents for each query, 
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save JSON data with mined hard negatives for finetuning
- `negative_number`: the number of sampled negatives 
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)**
- `candidate_pool`: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. 
The format of this file is the same as pretrain data which is jsonl {'text': str}. If input a candidate_pool, this script will retrieve negatives from this file.
- `use_gpu_for_searching`: whether to use faiss-gpu to retrieve negatives.

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

### Params Setting
- `--per_device_train_batch_size`: set batch size of each gpu, total_batch_size = n_gpu x per_device_train_batch_size
- `--normlized`: whether L2 norm in features
- `--query_max_len`: max length of query setting
- `--passage_max_len` 512: max length of passage setting

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