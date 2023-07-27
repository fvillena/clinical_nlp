#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)
    args = parser.parse_args()
    with open(args.prediction_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    true = [l.split('\t')[0] for l in lines]
    predicted = [l.split('\t')[1].strip() for l in lines]
    with open(args.report_file, 'w', encoding="utf-8") as f:
        f.write(
            json.dumps(
                classification_report(true, predicted, output_dict=True),
                indent=4,
                ensure_ascii=False
                ))