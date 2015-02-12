from collections import namedtuple


class DescentParameters(namedtuple('DescentParameters', ['train_loss',
                                                         'test_loss',
                                                         'epochs',
                                                         'minibatch',
                                                         'augmented',
                                                         'learning_rate',
                                                         'momentum',
                                                         'l1',
                                                         'l2',
                                                         'start_decay',
                                                         'tolerance',
                                                         'tries'])):
    """
    Class implementing the parameters for gradient descent
    Wrapper on namedtuple so we can provide default values to the parameters
    """

    def __new__(cls, train_loss='crossentropy', test_loss=None,
                epochs=100, minibatch=10, augmented=1, learning_rate=0.001, momentum=0.,
                l1=0.001, l2=0.001, start_decay=1000,
                tolerance=1e-4, tries=5):

        if test_loss == None:
            test_loss = train_loss

        return super(DescentParameters, cls).__new__(cls, train_loss, test_loss,
                                                     epochs, minibatch, augmented, learning_rate, momentum,
                                                     l1, l2, start_decay,
                                                     tolerance, tries)


class ValidationParameters(namedtuple('ValidationParameters', ['type',
                                                               'test_size',
                                                               'period'])):
    """
    Class implementing the parameters for cross validation
    Wrapper on named tuple so we can provide default values to the parameters
    """

    def __new__(cls, type='cv', test_size=0.2, period=None):
        return super(ValidationParameters, cls).__new__(cls, type, test_size, period)

