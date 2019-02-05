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

if __name__ == '__main__':
    clf = SVMClassifier('../dataset_generation/dataset/46_problem_24-01-2019_10-37-29.txt')
    X_train, X_test, y_train, y_test = clf.process_dataset()
    clf.set_model()
    clf.train(X_train, y_train)
    clf.test(X_test, y_test)