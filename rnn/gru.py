import os
import sys
import keras
import logging
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, GRU, Dropout, Embedding, Flatten

logging.basicConfig(level=logging.DEBUG, filename='logs/lstm.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def _start(gpu):
    # Set the gpu on tensorflow.

    import tensorflow as tf

    # Set gpu.
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)


_start(2)


class LSTM_model():
    """Build, train, and test LSTM model."""
    def __init__(self, dataset, vocabulary=3, hidden_size=64, dropout=0.5,
        n_classes=2, activation='softmax', loss='binary_crossentropy',
        optimizer='adam', epochs=10000, metrics=['accuracy']):
        sentence = """Instantiating LSTM class with the following arguments:
        dataset_path: %s; vocabulary_size: %d; hidden_size: %d; dropout: %.1f;
        n_classes: %d; activation: %s; loss: %s; optimizer: %s""" % (dataset,
            vocabulary, hidden_size, dropout, n_classes, activation, loss,
            optimizer)
        logging.debug(sentence)
        self.dataset =  dataset
        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_classes = n_classes
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.num_steps = None
        self.epochs = epochs

    def process_dataset(self):
        # Read data.
        data = open(self.dataset, 'r').readlines()

        # Remove copies.
        data = list(set(data))
        logging.debug("Read data and obtained %d unique samples." % len(data))
        
        # Turn input into lists.
        X = []
        Y = []
        for i, smpl in enumerate(data):
            x, y = smpl.split()
            X.append([])
            Y.append(int(y))
            for c in x:
                X[i].append(int(c))
        X = np.array(X)
        print X.shape
        self.num_steps = X.shape[1]
        Y = np.array(Y)
        Y = to_categorical(Y)

        logging.debug("X example: %s\ny example: %s" % (X[0], Y[0]))
        # Split into train, val, and test set.
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
            test_size=0.33, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
         test_size=0.5, random_state=42)
        logging.debug("Train samples: %d; Val samples: %d; Test samples: %d" %
            (X_train.shape[0], X_val.shape[0], X_test.shape[0]))

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

        return X_train, X_val, X_test, y_train, y_val, y_test

    def set_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocabulary, self.hidden_size,
            input_length=self.num_steps))
        self.model.add(GRU(self.hidden_size, return_sequences=True))
        # model.add(LSTM(hidden_size, return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(Flatten())
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation(self.activation))

        self.model.compile(loss=self.loss, optimizer=self.optimizer,
            metrics=self.metrics)
        self.model.summary()

    def train(self, X_train, X_val, y_train, y_val):
        # Set callback names.
        dataset_name = self.dataset.split('/')[-1]
        name_base, _ = os.path.splitext(dataset_name)

        #print X_train.shape
        #print y_train.shape
        #print X_train
        #print y_train
        #sys.exit(1)

        # Set callbacks
        hist = History()
        early = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        model_check = ModelCheckpoint('models/checkpoint_' + name_base + ".hdf5",
         monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        self.model.fit(X_train, y_train, batch_size=None, epochs=self.epochs,
            verbose=1, callbacks=[hist,early, model_check], 
            validation_data=(X_val, y_val), shuffle=False,
            steps_per_epoch=X_train.shape[1] / 32,
            validation_steps=X_val.shape[1] / 32)

    def test(self, X_test, y_test):
        pass


if __name__ == '__main__':
    lstm = LSTM_model(
        '../dataset_generation/dataset/46_problem_24-01-2019_10-37-29.txt')
    X_train, X_val, X_test, y_train, y_val, y_test = lstm.process_dataset()
    lstm.set_model()
    lstm.train(X_train, X_val, y_train, y_val)
    lstm.test(X_test, y_test)
