import logging
import sklearn.cross_validation as cv

logger = logging.getLogger("TRAINING")

def gradient_descent(model, x, y,
                     descent_parameters,
                     validation='holdout',
                     minibatch=1,
                     epochs=100,
                     holdout_period=9,
                     tolerance=1e-4):

    # generate validation sets if needed
    if validation == 'cv':
        x_train, x_val, y_train, y_val = cv.train_test_split((x, y), test_size=0.2)
    else:
        x_train = x
        y_train = y

    cost = 1e100
    n_batches = x.shape[0] // minibatch
    period = 1

    model.begin_training()

    for epoch in range(epochs):
        for batch in range(n_batches):
            model.train(i)
