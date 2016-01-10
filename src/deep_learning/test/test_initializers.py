import scipy.stats as scs
from numpy.testing import assert_array_less, assert_almost_equal

from deep_learning.units.initialization.constant_initializer import ConstantInitializer
from deep_learning.units.initialization.random_initializers import RandomInitializer

def test_constant():
    ci = ConstantInitializer("constant", 1, 2)
    assert_almost_equal([2], ci.create_shared().get_value())

def test_random():
    ri = RandomInitializer("random", 1, scs.uniform(0, 1))
    v = ri.create_shared()
    assert_array_less(v.get_value(), [1])
    assert_array_less([0], v.get_value())
