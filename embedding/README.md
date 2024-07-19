## Env Prepare
```shell
conda create -n nlpdk python=3.10
conda activate nlpdk

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c pytorch faiss-gpu=1.8.0 -y
conda install chardet=4.0.0 -y
pip install FlagEmbedding==1.2.9
```