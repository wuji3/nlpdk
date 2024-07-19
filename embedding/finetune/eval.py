import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from FlagEmbedding import FlagModel

import logging
from typing import Union
import yaml
import time

class SmartLogger:

    _Instance = None
    _Flag = False

    def __new__(cls, *args, **kwargs):
        if cls._Instance is None:
            cls._Instance = super().__new__(cls)
        return cls._Instance

    def __init__(self, filename = None, level: int = 1):
        if not self.__class__._Flag:
            self.__class__._Flag = True
            # logger -> Singleton
            self.file_logger = self.create_logger('file', 'file', filename=filename) if filename is not None else None
            self.console_logger = self.create_logger('console', 'console')
            self.level = level

    def log(self, msg: Union[str, dict]):
        if isinstance(msg, dict):
            self.file_logger.info(yaml.dump(msg, sort_keys=False, default_flow_style=False))

        else: self.file_logger.info(msg)

    def console(self, msg: Union[str, dict]):
        if isinstance(msg, dict):
            self.console_logger.info('\n'+str(yaml.dump(msg, sort_keys=False, default_flow_style=False)))

        else:
            self.console_logger.info(msg)

    def __call__(self, msg: Union[str, dict]):
        if self.file_logger is not None: self.log(msg)
        self.console(msg)

    def create_logger(self, name: str, kind: str, filename = None):
        assert kind in {'file', 'console'}
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        # format
        file_format = logging.Formatter(fmt='%(asctime)-20s%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if kind == 'file':
            handler = logging.FileHandler(filename=filename)
            handler.setFormatter(file_format)
            logger.addHandler(handler)
        elif kind == 'console':
            handler = logging.StreamHandler()
            handler.setFormatter(file_format)
            logger.addHandler(handler)
        return logger

logger = SmartLogger()
@dataclass
class Args:
    encoder: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )

    corpus_data: str = field(
        default="namespace-Pt/msmarco",
        metadata={'help': 'candidate passages'}
    )

    corpus_data_name: str = field(
        default="namespace-Pt/msmarco",
        metadata={'help': 'candidate passages'}
    )

    query_data: str = field(
        default="namespace-Pt/msmarco-corpus",
        metadata={'help': 'queries and their positive passages for evaluation'}
    )
    
    query_data_name: str = field(
        default="namespace-Pt/msmarco-corpus",
        metadata={'help': 'queries and their positive passages for evaluation'}
    )

    split: str = field(
        default="train",
        metadata={'help': 'train or test'}
    )

    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=128,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )

    roc_save_path: str = field(
        default='',
        metadata={'help': 'Save path to npy file which contains the baseline TPR and FPR.'}
    )

def index(model: FlagModel, corpus: datasets.Dataset, batch_size: int = 256, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        corpus_embeddings = model.encode_corpus(corpus["content"], batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(model: FlagModel, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries["query"], batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    
    
def evaluate(preds, 
             preds_scores, 
             labels, 
             roc_save_path = None,
             cutoffs=[1, 3, 5, 10, 100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    # AUC 
    pred_hard_encodings = []
    for pred, label in zip(preds, labels):
        pred_hard_encoding = np.isin(pred, label).astype(int).tolist()
        pred_hard_encodings.append(pred_hard_encoding)
    
    from sklearn.metrics import roc_auc_score, roc_curve, ndcg_score
    pred_hard_encodings1d = np.asarray(pred_hard_encodings).flatten() 
    preds_scores1d = preds_scores.flatten()
    auc = roc_auc_score(pred_hard_encodings1d, preds_scores1d)
    fpr, tpr, thresh = roc_curve(pred_hard_encodings1d, preds_scores1d)   
    if roc_save_path:
        fpr_tpr = np.stack([fpr, tpr], axis=0)
        np.save(roc_save_path, fpr_tpr)
    
    metrics['AUC@100'] = auc

    # nDCG
    for k, cutoff in enumerate(cutoffs):
        nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
        metrics[f"nDCG@{cutoff}"] = nDCG
            
    return metrics


def main():
    import time
    t0 = time.time()

    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    if args.query_data == 'namespace-Pt/msmarco-corpus':
        assert args.corpus_data == 'namespace-Pt/msmarco'
        eval_data = datasets.load_dataset("namespace-Pt/msmarco", split="dev")
        corpus = datasets.load_dataset("namespace-Pt/msmarco-corpus", split="train")
    else:
        #eval_data = datasets.load_dataset('json', data_files=args.query_data, split='train')
        #corpus = datasets.load_dataset('json', data_files=args.corpus_data, split='train')
        eval_data = datasets.load_dataset(path = args.query_data, name= args.query_data_name, split = args.split).remove_columns(['neg']).rename_column('pos', 'positive')
        corpus = datasets.load_dataset(path = args.corpus_data, name = args.corpus_data_name, split = args.split)
    
    logger(args)

    model = FlagModel(
        args.encoder, 
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
        use_fp16=args.fp16
    )
    
    faiss_index = index(
        model=model, 
        corpus=corpus, 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )
    
    scores, indices = search(
        model=model, 
        queries=eval_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus[indice]["content"])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["positive"])
        
    metrics = evaluate(retrieval_results, scores, ground_truths, args.roc_save_path)

    for k, v in metrics.items():
        metrics[k] = round(float(v), 4)
    logger(metrics)

    logger(f'Total consume: {(time.time() - t0)/3600} hours')

if __name__ == "__main__":
    main()
