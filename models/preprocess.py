import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_set(dataset):
    # Read, organize, and return X and Y from dataset.
    # Read data.
    df = pd.read_csv(dataset, sep=' ')

    # Set X and Y.
    print df
    print df['sample'][0]
    sys.exit(0)
    X = np.zeros((len(df), len(df['sample'][0])), dtype='int')
    Y = df['class'].astype('int')
    for i, row in df.iterrows():
        for j, c in enumerate(row['sample']):
            X[i][j] = int(c)
    Y = Y.to_numpy(dtype='int')

    return X, Y

def read_dataset(dataset_path, balanced=True):
    # Read data.
    df = pd.read_csv(dataset_path, sep=' ')
    
    # Remove copies.
    df = df.drop_duplicates()

    # Set X and Y.
    X = np.zeros((len(df), len(df['sample'][0])), dtype='int')
    Y = df['class'].astype('int')
    out_index = 0 # Dataframe indexes changed due to drop_duplicates.
    for i, row in df.iterrows():
        for j, c in enumerate(row['sample']):
            X[out_index][j] = int(c)
        out_index += 1
    Y = Y.to_numpy(dtype='int')

    if balanced:
        X, Y = balance_data(X, Y)

    return X, Y


def balance_data(X, Y):
    clss, counts = np.unique(Y, return_counts=True)
    n_samples = dict(zip(clss, counts))
    print "Balancing data, 0: %d; 1: %d" % (n_samples[0], n_samples[1])

    class_xs = []   # Set a list to receive the class and elements.
    min_elems = None    # Set the minimum elements to None.

    for yi in np.unique(Y):
        # Run over the unique classes in Y (0, 1).
        elems = X[(Y == yi)]   # Get X samples from the selected class.
        class_xs.append((yi, elems)) # Add the class and the elements.
        if min_elems == None or elems.shape[0] < min_elems:
            # Set the minimum number of elements in elems set.
            min_elems = elems.shape[0]

    # Set the new list of balanced elements.
    xs = []
    ys = []

    for ci, this_xs in class_xs:        
        x_ = this_xs[:min_elems] # Fill X.
        y_ = np.empty(min_elems) # Set y size.
        y_.fill(ci) # Fill with the class.
        xs.append(x_) 
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


def split_dataset(X, Y, test_size=0.33, validation=True):
    
    # Split into train, val, and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
        test_size=test_size, random_state=42)
    # X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])

    if validation:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
            test_size=0.5, random_state=42)
        # X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1])
        # X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        return X_train, X_test, y_train, y_test
    
    
