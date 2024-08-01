from FlagEmbedding import FlagReranker
import argparse
import datasets
import os
from typing import Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-m', '--model_name_or_path', type=str, default='/home/duke/nlpdk/rerank/rerank_output')
    args.add_argument('-d', '--data', type=str, default='shoppal/embedding-index', help='Dir, File & HF dataset are ok')
    args.add_argument('-sd', '--sub_data_name', type=Optional[str], default=None, help='Sub_data_name if huggingface dataset')

    return args.parse_args() 

def main(args):
    # for debugging
    # args.sub_data_name = 'source2.3'

    reranker = FlagReranker(args.model_name_or_path, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    if os.path.isdir(args.data):
        train_datasets = []
        for file in os.listdir(args.data):
            temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.data, file),
                                                    split='train')
            train_datasets.append(temp_dataset)
        dataset = datasets.concatenate_datasets(train_datasets)
    # huggingface hub dataset
    elif args.sub_data_name is not None:
        dataset = datasets.load_dataset(args.data, name=args.sub_data_name, split='test')
    # local data json file
    else:
        dataset = datasets.load_dataset('json', data_files=args.data, split='test')
    
    def process(examples):
        query = examples['query']
        
        positive_pair, negative_pair = [], []
        for q, pos in list(zip(query, examples['pos'])):
            for p in pos:
                positive_pair.append([q, p])
        
        for q, neg in list(zip(query, examples['neg'])):
            for n in neg:
                negative_pair.append([q, n])
            
        positive_label, negative_label = [1] * len(positive_pair), [0] * len(negative_pair)

        eval_dataset['pairs'] += positive_pair + negative_pair
        eval_dataset['labels'] += positive_label + negative_label
    
    eval_dataset = {'pairs': [], 'labels': []}
    dataset.map(process, batched=True)
    
    scores = reranker.compute_score(eval_dataset['pairs'], normalize=True)
    prediction = list(map(round, scores))
    
    precision, recall, f1  = precision_score(eval_dataset['labels'], prediction), recall_score(eval_dataset['labels'], prediction), f1_score(eval_dataset['labels'], prediction)
    auc = roc_auc_score(eval_dataset['labels'], scores)
    print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}')

if __name__ == '__main__':
    main(parse_args())