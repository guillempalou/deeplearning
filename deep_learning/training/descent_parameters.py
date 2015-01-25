from collections import namedtuple


# define the learning parameters for gradient descent
DescentParameters = namedtuple('DescentParameters', ['learning_rate',
                                                     'momentum',
                                                     'l1',
                                                     'l2',
                                                     'decay'])


