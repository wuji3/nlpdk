## Env Prepare
```shell
conda create -n nlp-rerank python=3.10 -y
conda activate nlp-rerank

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c pytorch faiss-gpu=1.8.0 -y
conda install chardet=4.0.0 -y
pip install FlagEmbedding==1.2.9
pip install sentencepiece==0.2.0
```

## Data Prepare
1. Data For Training Like [Embedding](../embedding/README.md)
    ```json
    {
        "query": "Do wireless headsets have static noise?", 
        "pos": ["If you have a wireless headset, it's possible that static is caused by interference from nearby electronics. Walk around your home or office and listen to the headset to see if your static disappears. Sometimes a problem with software is what's causing static.", "..."],
        "neg": ["Downloaded is None in fetch_url(https:\/\/www.simplyheadsets.com.au\/blog\/how-to-fix-static-coming-from-your-headset#:~:text=If%20you%20have%20a%20wireless,software%20is%20what's%20causing%20static.) function", "..."]
    }
    ```

## OHEM
```bash
cd ../embedding & \
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

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run.py \
    --model_name_or_path BAAI/bge-reranker-v2-m3 \
    --train_data {realpath to jsonl file} \
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
    ```
2. If some json files in xxx directory

    ```bash
    #! /bin/bash

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run.py \
    --model_name_or_path BAAI/bge-reranker-v2-m3 \
    --train_data {realpath to jsonl dir} \
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
    ```
### Huggingface Data

    ```bash

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 run.py \
    --model_name_or_path BAAI/bge-reranker-v2-m3 \
    --train_data shoppal/embedding-index \
    --sub_data_name source2.3 \
    --learning_rate 6e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --dataloader_drop_last True \
    --train_group_size 4 \
    --max_len 513 \
    --weight_decay 0.01 \
    --logging_steps 500 \
    --save_steps 500 \
    --overwrite_output_dir True \
    ```

### Params Setting
- `--per_device_train_batch_size`: set batch size of each gpu, total_batch_size = n_gpu x per_device_train_batch_size
- `--max_len`: max length of passage setting, bge-reranker-v2-m3 support 8192 length.

## Eval

```python
python eval.py -m xxx -d {dir or file or huggingface} {-sd (if huggingface and has sub_data_name)}

```
## Infer
```markdown
Refer to eval.py
```