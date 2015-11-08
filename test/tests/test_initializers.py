import scipy.stats as scs
from deep_learning.initialization.random_initializers import RandomInitializer
from numpy.testing import assert_array_less, assert_almost_equal

from deep_learning.units.initialization.constant_initializer import ConstantInitializer


def test_constant():
    ci = ConstantInitializer("constant", shape=1, value=2)
    assert_almost_equal([2], ci().get_value())

def test_random():
    ri = RandomInitializer("uniform", d, distribution=scs.uniform(0, 1))
    v = ri()
    assert_array_less(v.get_value(), [1])
    assert_array_less([0], v.get_value())
