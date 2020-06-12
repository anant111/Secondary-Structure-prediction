
import numpy as np
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset, split_with_shuffle, get_data_labels, split_like_paper, get_cb513
import model
from time import time

do_log = True
stop_early = False
show_plots = True

dataset = get_dataset()

D_train, D_test, D_val = split_with_shuffle(dataset, 100)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

net = model.CNN_model()

start_time = timer()

history = None

call_b = [model.checkpoint]

if do_log:
    call_b.append(callbacks.TensorBoard(log_dir="../logs/Whole_CullPDB/{}".format(time()), histogram_freq=0, write_graph=True))

if stop_early:
    call_b.append(model.early_stop)

history = net.fit(X_train, Y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=call_b)

end_time = timer()
print("\n\nTime elapsed: " + "{0:.2f}".format((end_time - start_time)) + " s")

predictions = net.predict(X_test)

print("\n\nQ8 accuracy: " + str(model.Q8_accuracy(Y_test, predictions)) + "\n\n")

CB513_X, CB513_Y = get_cb513()

predictions = net.predict(CB513_X)

print("\n\nQ8 accuracy on CB513: " + str(model.Q8_accuracy(CB513_Y, predictions)) + "\n\n")

if show_plots:
    from plot_history import plot_history
    plot_history(history)

