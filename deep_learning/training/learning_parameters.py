from collections import namedtuple


class DescentParameters(namedtuple('DescentParameters', ['epochs',
                                                         'minibatch',
                                                         'learning_rate',
                                                         'momentum',
                                                         'l1',
                                                         'l2',
                                                         'decay',
                                                         'tolerance',
                                                         'tries'])):
    """
    Class implementing the parameters for gradient descent
    Wrapper on namedtuple so we can provide default values to the parameters
    """

    def __new__(cls, epochs=100, minibatch=10, learning_rate=0.001, momentum=0., l1=0.001, l2=0.001, decay=0.,
                tolerance=1e-4, tries=3):
        return super(DescentParameters, cls).__new__(cls, epochs, minibatch, learning_rate, momentum, l1, l2, decay,
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

