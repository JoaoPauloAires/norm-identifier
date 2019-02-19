import os
import sys
import keras
import logging
import argparse
import preprocess
import numpy as np
import matplotlib.pyplot as plt
from keras import metrics
from pylab import savefig
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras.layers import Dense, Activation, GRU, Dropout, Embedding, Flatten

GPU_DEFAULT = 0

if not os.path.isdir('./logs'):
    os.mkdir('./logs')

logging.basicConfig(level=logging.DEBUG, filename='logs/lstm.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def _start(gpu):
    # Set the gpu on tensorflow.
    import tensorflow as tf

    # Set gpu.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


class RNN_model():
    """Build, train, and test LSTM model."""
    def __init__(self, dataset, vocabulary=2, hidden_size=64, dropout=0.5,
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
        
        X, Y = preprocess.read_dataset(self.dataset, balanced=True)
        print X.shape 

        self.num_steps = X.shape[1]
        Y = to_categorical(Y)

        logging.debug("X example: %s\ny example: %s" % (X[0], Y[0]))
        
        return preprocess.split_dataset(X, Y)

    def set_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocabulary, self.hidden_size,
            input_length=self.num_steps))
        self.model.add(GRU(self.hidden_size))#, return_sequences=True))
        # model.add(LSTM(hidden_size, return_sequences=True))
        self.model.add(Dropout(self.dropout))
        # self.model.add(Flatten())
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation(self.activation))

        self.model.compile(loss=self.loss, optimizer=self.optimizer,
            metrics=self.metrics)
        self.model.summary()

    def train(self, X_train, X_val, y_train, y_val, plot=False):
        # Set callback names.
        dataset_name = self.dataset.split('/')[-1]
        name_base, _ = os.path.splitext(dataset_name)

        # Set callbacks.
        hist = History()
        early = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        if not os.path.isdir('saved_models'):
            os.mkdir('./saved_models')
        model_check = ModelCheckpoint(
            'saved_models/rnn_checkpoint_' + name_base + ".hdf5",
            monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        history = self.model.fit(X_train, y_train, batch_size=None,
            epochs=self.epochs, verbose=1,
            callbacks=[hist, early, model_check], 
            validation_data=(X_val, y_val), shuffle=False,
            steps_per_epoch=X_train.shape[0] / 16,
            validation_steps=X_val.shape[0] / 16)

        if plot:
            if not os.path.isdir('./plots'):
                os.mkdir('./plots')
            model_feat = "hs-%d_dp-%.1f_actv-%s_opt-%s" % (self.hidden_size,
                self.dropout, self.activation, self.optimizer)
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plot_name = "plots/model_accuracy_" + model_feat
            savefig(plot_name + '.pdf', bbox_inches='tight')
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plot_name = "plots/model_loss_" + model_feat
            savefig(plot_name + '.pdf', bbox_inches='tight')

    def test(self, X_test, y_test):
        dataset_name = self.dataset.split('/')[-1]
        name_base, _ = os.path.splitext(dataset_name)
        self.model.load_weights(
            'saved_models/rnn_checkpoint_' + name_base + ".hdf5")
        print self.model.evaluate(x=X_test, y=y_test)
        w_file = open("pred_" + name_base + ".txt", 'w')
        w_file.write("pred true\n")
        for ind, x_smp in enumerate(X_test):
            x_smp = x_smp.reshape(1, x_smp.shape[0])
            pred = np.argmax(self.model.predict(x_smp))
            true = np.argmax(y_test[ind])
            w_file.write("%d %d\n" % (pred, true))


def run_gru(dataset):
    gru = RNN_model(dataset, metrics=[metrics.binary_accuracy])
    X_train, X_val, X_test, y_train, y_val, y_test = gru.process_dataset()
    gru.set_model()
    gru.train(X_train, X_val, y_train, y_val)
    gru.test(X_test, y_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GRU.')
    parser.add_argument('env_name', type=str, help="Environment name.")
    parser.add_argument('-d', '--dataset', type=str, help="Path to dataset.")
    parser.add_argument('-f', '--folder', type=str,
        help="Folder to datasets.")
    parser.add_argument('--gpu', type=int, help="Indicate the gpu number.")

    args = parser.parse_args()
    if args.gpu:
        _start(args.gpu)
    else:
        _start(GPU_DEFAULT)
    if args.dataset:
        run_gru(args.dataset)
    elif args.folder:
        folder_path = args.folder
        files = os.listdir(folder_path)
        for f in files:
            if args.env_name not in f:
                continue
            obs_path = os.path.join(folder_path, f)
            print obs_path
            run_gru(obs_path)

    # n-276_con-368_car-51_ob-36_enf-98
