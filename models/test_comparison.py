import os
import sys
import gru
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score


def train_files(folder_path):
    folder_struc = dict()
    files = os.listdir(folder_path)
    return_files = []
    for f in files:
        file_id = f[0]
        file_type = f.split('_')[1].split('.')[0]
        if file_id not in folder_struc:
            folder_struc[file_id] = dict()
        folder_struc[file_id][file_type] = os.path.join(folder_path, f)
    for k in folder_struc:
        train_path = folder_struc[k]['train']
        test_path = folder_struc[k]['test']
        return_files.append(gru.run_gru(train=train_path, test=test_path))

    return return_files


def evaluate(files):

    for f in files:
        df = pd.read_csv(f, sep=' ')
        y_pred = df["pred"]
        y_true = df["true"]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print "%s results:\n\tAcc: %.2f\n\tPrec: %.2f\n\tRec: %.2f\n\tF1: %.2f" % (
            f, acc, prec, rec, f1)


def main(folder_path):
    files = train_files(folder_path)
    evaluate(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test comparison.')
    parser.add_argument('folder_path', type=str, help="Path to test.")

    args = parser.parse_args()
    main(args.folder_path)