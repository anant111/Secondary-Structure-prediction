import numpy as np
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset, split_with_shuffle, get_data_labels, split_like_paper, get_cb513
import model
dataset = get_dataset()
np.set_printoptions(threshold=np.inf)

D_train, D_test, D_val = split_with_shuffle(dataset, 100)

X_test, Y_test = get_data_labels(D_test)

print(X_test[2], file=open("output.txt", "a"))
