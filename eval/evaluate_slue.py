import argparse
import logging

from progress.bar import Bar

import json
import pandas as pd

from metrics import ErrorMetric
from util import format_results, load_predictions, load_gold_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)


def load_json2dict(file_name):
    wrong_dict_sample_num = 0
    res = {}
    with open(file_name, encoding='utf-8') as f:
        sample_lines = f.readlines()
        total_sample = len(sample_lines)
        for i in range(int(total_sample)):
            sample = json.loads(sample_lines[i])
            if sample['ID'] not in res.keys():
                res[sample['ID']] = {}
                try:
                    res[sample['ID']]['entities'] = eval(sample['entities'].replace('|', ','))
                except:
                    print(sample)
                    res[sample['ID']]['entities'] = [{'type': 'None', 'filler': 'None'}]
                    wrong_dict_sample_num += 1
            else:
                raise ValueError('The ID has been overlapped')
    return res, wrong_dict_sample_num

def load_csv2dict(file_name):
    res = {}
    df = pd.read_csv(file_name, header=0, sep=',')
    for index, sample in df.iterrows():
        if sample['ID'] not in res.keys():
            res[str(sample['ID'])] = {}
            res[str(sample['ID'])]['entities'] = eval(sample['semantics'].replace('|', ','))
        else:
            raise ValueError('The ID has been overlapped')
    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLURP evaluation script')
    parser.add_argument(
        '-g',
        '--gold-data',
        required=True,
        type=str,
        help='Gold data in SLURP jsonl format'
    )
    parser.add_argument(
        '-p',
        '--prediction-file',
        type=str,
        required=True,
        help='Predictions file'
    )
    parser.add_argument(
        '--load-gold',
        action="store_true",
        help='When evaluating against gold transcriptions (gold_*_predictions.jsonl), this flag must be true.'
    )
    parser.add_argument(
        '--average',
        type=str,
        default='micro',
        help='The averaging modality {micro, macro}.'
    )
    parser.add_argument(
        '--full',
        action="store_true",
        help='Print the full results, including per-label metrics.'
    )
    parser.add_argument(
        '--errors',
        action="store_true",
        help='Print TPs, FPs, and FNs in each row.'
    )
    parser.add_argument(
        '--table-layout',
        type=str,
        default='fancy_grid',
        help='The results table layout {fancy_grid (DEFAULT), csv, tsv}.'
    )

    args = parser.parse_args()

    logger.info("Loading data")
    # pred_examples = load_predictions(args.prediction_file, args.load_gold)
    # gold_examples = load_gold_data(args.gold_data, args.load_gold)
    pred_examples, wrong_dict_sample_num = load_json2dict(args.prediction_file)
    gold_examples = load_csv2dict(args.gold_data)
    n_pred_examples = len(pred_examples)
    n_gold_examples = len(gold_examples)

    logger.info("Initializing metrics")
    # scenario_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    # action_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    # intent_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    span_f1 = ErrorMetric.get_instance(metric="span_f1", average=args.average)
    distance_metrics = {}
    for distance in ['word', 'char']:
        distance_metrics[distance] = ErrorMetric.get_instance(metric="span_distance_f1",
                                                              average=args.average,
                                                              distance=distance)
    slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=args.average)

    bar = Bar(message="Evaluating metrics", max=len(gold_examples))
    wrong_keyword_sample_num = 0
    for gold_id in list(gold_examples):
        if gold_id in pred_examples:
            gold_example = gold_examples.pop(gold_id)
            pred_example = pred_examples.pop(gold_id)

            # below is added by a person
            flag = False
            if "entities" not in pred_example.keys():
                flag = True
                pred_example["entities"] = [{'type': 'None', 'filler': 'None'}]
                # pred_example["entities"] = []
            if flag == True:
                print(pred_example.keys())
                wrong_keyword_sample_num += 1

            # above is added by a person

            span_f1(gold_example["entities"], pred_example["entities"])
            for distance, metric in distance_metrics.items():
                metric(gold_example["entities"], pred_example["entities"])
        bar.next()
    bar.finish()

    logger.info("Results:")


    results = span_f1.get_metric()
    print(format_results(results=results,
                         label="entities",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

    for distance, metric in distance_metrics.items():
        results = metric.get_metric()
        slu_f1(results)
        print(format_results(results=results,
                             label="entities (distance {})".format(distance),
                             full=args.full,
                             errors=args.errors,
                             table_layout=args.table_layout), "\n")
    results = slu_f1.get_metric()
    print(format_results(results=results,
                         label="SLU F1",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

    logger.warning("Gold examples not predicted: {} (out of {})".format(len(gold_examples), n_gold_examples))
    logger.warning("Keyword-mistake examples in predicted: {} (out of {})".format(wrong_keyword_sample_num, n_pred_examples))
    logger.warning(
        "dict-mistake examples in predicted: {} (out of {})".format(wrong_dict_sample_num, n_pred_examples))
