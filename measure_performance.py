#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)
    args = parser.parse_args()
    if args.prediction_file.endswith(".txt"):
        from sklearn.metrics import classification_report
        with open(args.prediction_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        true = [l.split("\t")[0] for l in lines]
        predicted = [l.split("\t")[1].strip() for l in lines]
        report = classification_report(true, predicted, output_dict=True)
    elif args.prediction_file.endswith(".conll"):
        from seqeval.metrics import classification_report
        true = []
        predicted = []
        with open(args.prediction_file, "r", encoding="utf-8") as f:
            current_true = []
            current_predicted = []
            for line in f:
                if len(line.strip()) == 0:
                    true.append(current_true)
                    current_true = []
                    predicted.append(current_predicted)
                    current_predicted = []
                else:
                    _, t, p = line.strip().split("\t")
                    current_true.append(t)
                    current_predicted.append(p)
        report = classification_report(true, predicted, output_dict=True)
    with open(args.report_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=4, ensure_ascii=False, cls=NpEncoder))
