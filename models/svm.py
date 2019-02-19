import os
import argparse
import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score


class SVMClassifier(object):
    """SVM_Class"""
    def __init__(self, dataset, model=None):
        self.dataset = dataset
        self.model = model

    def process_dataset(self):
        X, Y = preprocess.read_dataset(self.dataset, balanced=True)
        return preprocess.split_dataset(X, Y, validation=False)

    def set_model(self, C=1.0, kernel='rbf', gamma='auto'):
        if not self.model:
            self.model = svm.SVC(C=C, kernel=kernel, gamma=gamma)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print "Acc: %.2f; Prec: %.2f; Recall: %.2f; F1_score: %.2f" % (
            acc, prec, rec, f1)
        return acc, prec, rec, f1


def run_svm(dataset):
    clf = SVMClassifier(dataset)
    X_train, X_test, y_train, y_test = clf.process_dataset()
    clf.set_model()
    clf.train(X_train, y_train)
    return clf.test(X_test, y_test)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train SVM.')
    parser.add_argument('env_name', type=str, help="Environment name.")
    parser.add_argument('-d', '--dataset', type=str, help="Path to dataset.")
    parser.add_argument('-f', '--folder', type=str,
        help="Folder to datasets.")

    args = parser.parse_args()

    if args.dataset:
        acc, prec, rec, f1 = run_svm(args.dataset)
        
    elif args.folder:
        folder_path = args.folder
        files = os.listdir(folder_path)
        mean_acc, mean_prec, mean_rec, mean_f1 = 0, 0, 0, 0
        for f in files:
            if args.env_name not in f:
                continue
            obs_path = os.path.join(folder_path, f)
            print obs_path
            acc, prec, rec, f1 = run_svm(obs_path)
            mean_acc += acc
            mean_prec += prec
            mean_rec += rec
            mean_f1 += f1
        n_dataset = len(files)
        mean_acc = mean_acc/float(n_dataset)
        mean_prec = mean_prec/float(n_dataset)
        mean_rec = mean_rec/float(n_dataset)
        mean_f1 = mean_f1/float(n_dataset)
        print "Mean Acc: %.2f; Mean Prec: %.2f; Mean Rec: %.2f; Mean F1: %.2f" % (mean_acc, mean_prec, mean_rec, mean_f1)
