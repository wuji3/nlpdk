from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import argparse
from utils.logger import SmartLogger
from tqdm import tqdm


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--inp', type=str, default="product", help='Field as input of model')
    argparser.add_argument('-tk', '--topk', type=int, default=5, help='Prediction topk')
    argparser.add_argument('-dv', '--device', type=str, default='cuda:0', help='GPU device')
    argparser.add_argument('-od', '--outdir', type=str, default='/home/wuji3/nlpdk/test', help='Output dir')
    argparser.add_argument('-f', '--predictf', type=str, default='/home/wuji3/dkdata/iter1data/train_debug.csv', help='Which file to predict')
    argparser.add_argument('-fi', '--field', nargs='+', default=[], help='Field with sentence1')
    argparser.add_argument('-m', '--model', type=str, default='/home/wuji3/nlpdk/test/checkpoint-48')
    argparser.add_argument('-bs', '--bs', type=int, default=512, help='Batch size of predicting')

    return argparser.parse_args()

def main(args):
    # log
    logger = SmartLogger()

    os.makedirs(args.outdir, exist_ok=True)

    # args
    topk = args.topk
    device = args.device
    output_dir = args.outdir
    predict_file = args.predictf
    task = os.path.basename(predict_file).split('.')[0]
    output_predict_file = os.path.join(output_dir, f"results_predict_{task}.csv")
    model_path = args.model
    batch_size = args.bs

    # model
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=config).to(device)

    # data tokenize
    raw_df = pd.read_csv(predict_file, dtype={args.inp: 'str'})
    raw_df.rename(columns={args.inp: "sentence1"}, inplace = True)
    df = raw_df[raw_df['sentence1'].apply(lambda x: isinstance(x, str))].copy()
    text = df['sentence1'].tolist()

    with torch.inference_mode():

        output_probs = []
        for i in tqdm(range(0, len(text), batch_size), desc="Infer ..."):
            batch = tokenizer(text[i: i+batch_size], padding=True, truncation=True, max_length=config.max_position_embeddings,  return_tensors='pt').to(device)
            # batch = {k: v[i:i+batch_size] for k, v in inputs.items()}
            batch_output = model(**batch)

            logits = batch_output.logits
            if config.problem_type == 'single_label_classification':
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            elif config.problem_type == 'multi_label_classification':
                probs = F.sigmoid(logits).detach().cpu().numpy()
            else: 
                raise ValueError('problem_type must be single-label-classification or multi-label-classification')
            output_probs.append(probs)
        
        probs = np.concatenate(output_probs, axis=0)

    if config.num_labels < topk: 
        logger(f'Class number {config.num_labels} < Topk {topk}, so predict all classes.')
    topk = min(topk, config.num_labels)

    pl = np.argsort(probs, axis=1)[:, ::-1][:, :topk]

    # Prepare the data
    data = []
    for index, item in tqdm(enumerate(pl), desc='Mapping Label ...'):
        topk_item = [config.id2label[label_idx] for label_idx in item]
        row = [text[index]] + topk_item + [[str(round(p, 2)) for p in probs[index][item].tolist()]]
        data.append(row)

    df = pd.DataFrame(data, columns=["sentence1"] + [f"top{i+1}" for i in range(topk)] + ['score'])
    fields = ['sentence1'] + [*args.field]
    df = pd.merge(df, raw_df[fields], how='left', on='sentence1')
    df.to_csv(output_predict_file, index=False)

if __name__ == '__main__':
    main(parse_args())