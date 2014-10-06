""" Basic form to follow while writing nosetest tests. """

from nose import with_setup

def setup_func():
    print "setup..."

def teardown_func():
    print "teardown..."

@with_setup(setup_func, teardown_func)
def test_squared_loss():
    for i in range(0, 9):
        yield trivial, i, i

def trivial(a, b):
    assert a == b

