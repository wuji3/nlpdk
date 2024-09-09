"""
This script are uesd to output the prediction analysis result of the model. 
Two level file will be outputed: Category-level and Case-level.

Category Level: such as precision, recall, f1score, test_count, train_count of each category.
Case Level: such as product, label, prediction.

"""
from tqdm import tqdm
import pandas as pd
import argparse
from utils.writer import ShardByShardWriter
from utils.logger import SmartLogger

def parse_opt():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('-s', '--source_data', default='/home/duke/dkdata/iter1data/val.csv', help='Source data to be predicted')
    parsers.add_argument('-p', '--prediction_data', default='predict_results_None.csv', type=str, help='Prediction from source data')
    parsers.add_argument('-caseout', '--bycase_output', default='prediction_analysis_bycase.csv', help='Output of prediction_analysis_bycase')
    parsers.add_argument('-f', '--field', nargs='+', default=[], help='Field with sentence1')
    parsers.add_argument('-td', '--train_data', default='train.csv', help='Get the training data numbers of each category')
    parsers.add_argument('-cateout', '--bycategory_output', default='prediction_analysis_bycategory.csv', help='Output of prediction_analysis_bycategory')
    parsers.add_argument('-tk', '--topk', default=5, type=int, help='Topk of prediction')

    return parsers.parse_args()

def main(args):

    # Logger
    logger = SmartLogger()

    # Generate bycase.csv containing [sentence1, label, top1, top2, ..., score]
    predict_data = pd.read_csv(args.prediction_data)
    roi_field = ['sentence1', 'label'] + args.field
    source_data = pd.read_csv(args.source_data)[roi_field]
    
    merge_data = pd.merge(predict_data, source_data, on='sentence1')
    
    merge_data = merge_data[roi_field + [f'top{i+1}'for i in range(args.topk)] + ['score']]

    # Compute acc1 and acctopk
    merge_data['acc1'] = merge_data['label'] == merge_data['top1']
    merge_data[f'acc{args.topk}'] = merge_data.apply(lambda x: x['label'] in x[[f'top{i+1}' for i in range(args.topk)]].values, axis=1)

    acc1, acctopk = merge_data['acc1'].mean(), merge_data[f'acc{args.topk}'].mean()
    logger(f"acc1: {acc1}, acc{args.topk}: {acctopk}")
    
    merge_data.drop(labels=['acc1', f'acc{args.topk}'], axis=1, inplace=True)

    merge_data.to_csv(args.bycase_output, index=False)
    logger(f'Bycase output Done. Saved to {args.bycase_output}')
 
    # Generate bycategory.csv containing [label, precision, recall, f1score, test_count, train_count]
    pencile = ShardByShardWriter(ftype='csv')

    train_data = pd.read_csv(args.train_data)[['sentence1', 'label']]

    labels: list = merge_data['label'].unique().tolist()

    epison = 1e-5
    mp, mr, mf1 = 0, 0, 0
    for label in tqdm(labels, desc='Anaysize Prediction...', total=len(labels)):
        TP_FN = merge_data[merge_data['label'] == label]
        TP_FP = merge_data[merge_data['top1'] == label]
        TP = merge_data[(merge_data['label'] == merge_data['top1']) & (merge_data['label'] == label)]
        precision = TP.values.shape[0] / (TP_FP.values.shape[0] + epison)
        recall = TP.values.shape[0] / (TP_FN.values.shape[0] + epison)
        f1score = 2 * precision * recall / (precision + recall + epison)

        oneline = {'label': [label], 'precision': [precision], 'recall': [recall], 'f1score': [f1score], 'test_count': [TP_FN.values.shape[0]], 'train_count': [(train_data["label"] == label).sum()]}
        onedf = pd.DataFrame.from_dict(oneline)
        pencile.write_shard2csv(onedf, args.bycategory_output)

        mp += precision
        mr += recall
        mf1 += f1score
    
    logger(f'Bycategory output Done. Save to {args.bycategory_output}')

    mp /= len(labels)
    mr /= len(labels)
    mf1 /= len(labels)
    
    logger(f'\nmprecision -> {mp}\nmrecall -> {mr}\nmf1score -> {mf1}')

if __name__ == '__main__':
    main(parse_opt())
    