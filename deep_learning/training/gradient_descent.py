import logging
import sklearn.cross_validation as cv

logger = logging.getLogger("TRAINING")


def gradient_descent(model, x, y, descent, validation):
    """
    Gradient descent algorithm
    :param model: model to train
    :param x: array with all the examples
    :param y: array with groundtruth. y.shape[0] == x.shape[0]
    :param descent: DescentParameters structure
    :param validation: ValidationParameters structure
    :return: yields tuple of costs for train and test
    """

    # generate validation sets if needed
    if validation.type == 'cv':
        x_train, x_val, y_train, y_val = cv.train_test_split(x, y, test_size=0.2)
        logger.debug("Dividing dataset into {0} for train and {1} for validation".format(x_train.shape[0],
                                                                                         x_val.shape[0]))
    else:
        x_train = x
        y_train = y
        # they shouldn't be used in this case
        x_val = None
        y_val = None

    cost = 1e100

    tries = 0
    n_batches = x_train.shape[0] // descent.minibatch
    period = 1

    # set up the training for the model

    logger.info("Beginning training for model {0}".format(model.name))
    logger.debug("Learning parameters {0}".format(descent))
    logger.debug("Validation parameters {0}".format(validation))

    model.begin_training(descent, x=x_train, y=y_train)

    logger.info("Beginning epochs")
    for epoch in range(descent.epochs):
        current_cost = 0
        n_test = 0
        train_cost = 0
        for batch in range(n_batches):
            if validation.type == 'holdout' and period % validation.period == 0:
                test_cost = model.test(index=batch)
                current_cost += test_cost
                n_test += 1
                yield train_cost, test_cost
            else:
                train_cost += model.train(index=batch) / n_batches

        if validation.type == 'cv':
            current_cost = model.test(x=x_val, y=y_val)
            n_test = 1
            yield train_cost, current_cost

        current_cost /= n_test
        rel_improvement = (cost - current_cost) / cost
        cost = current_cost

        logger.info("Epoch {0} - {1} mean train error and {2} mean test error".format(epoch, train_cost, current_cost))
        logger.debug("Relative cost descrease: {0}".format(rel_improvement))

        if rel_improvement < descent.tolerance:
            tries += 1
        else:
            tries = 0

        if tries == descent.tries:
            logger.info("Training finished")
            break