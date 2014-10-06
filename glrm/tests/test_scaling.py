from numpy.random import rand
from numpy.testing import assert_array_almost_equal
from numpy import round, expand_dims, ones, zeros
from glrm.utils.fcn_utils import scale

def test_scaling():
    
    A = round(4*rand(40,60))
    missing = [(0,0), (2,2), (1,1), (3,3)]

    yield compare_scaling, A, missing, False
    yield compare_scaling, A, missing, True

def compare_scaling(A, missing, T):

    # automatic
    @scale(A, T, missing)
    def loss(P, V): return 0.5*(A - P.dot(V))**2

    # by hand
    A_filter = ones((A.shape))
    for indx in missing: A_filter[indx] = 0
    if not T:
        zero_col = expand_dims((0.5*A**2*A_filter).mean(0),0)/A_filter.mean(0)
        zero_col = ones((A.shape[0], 1)).dot(zero_col)
        means_col = expand_dims((0.5*((A - zero_col)**2)*A_filter).mean(0),
                0)/A_filter.mean(0)
        A_filter = A_filter/means_col

    def loss2(P, V): return (0.5*A_filter*(A - P.dot(V))**2).sum()

    # test
    P, V = ones((A.shape[0],1)), ones((1,A.shape[1]))
    print loss(P, V), loss2(P, V)
    assert loss(P, V) == loss2(P, V)
