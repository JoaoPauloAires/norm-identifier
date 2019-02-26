import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score


def calculate_metrics(results_path):
    mean_acc = 0
    mean_prec = 0
    mean_recall = 0
    mean_f1 = 0

    # Get folder files.
    files = os.listdir(results_path)
    n_obs = len(files)
    for fi in files:
        print fi
        # Read file content.
        file_path = os.path.join(results_path, fi)
        df = pd.read_csv(file_path, sep=' ')
        pred = df['pred'].to_numpy()
        true = df['true'].to_numpy()

        # Calculate metrics for this observer.
        acc = accuracy_score(true, pred)
        prec = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)

        print acc, prec, recall, f1 

        # Update mean values.
        mean_acc += acc
        mean_prec += prec
        mean_recall += recall
        mean_f1 += f1

    # Calculate the mean.
    mean_acc = mean_acc/float(n_obs)
    mean_prec = mean_prec/float(n_obs)
    mean_recall = mean_recall/float(n_obs)
    mean_f1 = mean_f1/float(n_obs)

    print "Average results:\nMean Accuracy: %.2f;\nMean Precision: %.2f;\nMean Recall: %.2f;\nMean F1: %.2f" % (
        mean_acc, mean_prec, mean_recall, mean_f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics.')
    parser.add_argument('results_path', type=str,
        help='Path to a folder containing the results for each observer.')

    args = parser.parse_args()
    calculate_metrics(args.results_path)
    