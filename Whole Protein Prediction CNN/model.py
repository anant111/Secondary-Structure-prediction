
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization
from keras import optimizers, callbacks
from keras.regularizers import l2
import tensorflow as tf

import dataset

do_summary = True

LR = 0.0005
drop_out = 0.3
batch_dim = 64
nn_epochs = 20

loss = 'categorical_crossentropy'


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

filepath="Whole_CullPDB-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


def Q8_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  
        for j in range(real.shape[1]): 
            if np.sum(real[i, j, :]) == 0:  
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    return correct / total


def CNN_model():
   
    m = Sequential()
    m.add(Conv1D(128, 11, padding='same', activation='relu', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))  
    m.add(Dropout(drop_out))  
    m.add(Conv1D(64, 11, padding='same', activation='relu'))  
    m.add(Dropout(drop_out))  
    m.add(Conv1D(dataset.num_classes, 11, padding='same', activation='softmax'))  
    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")

        m.summary()

    return m


if __name__ == '__main__':
    print("This script contains the model")
