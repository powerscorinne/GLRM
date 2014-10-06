from glrm import categorical_loss
from numpy.random import rand
from numpy import round, ones
from numpy.testing import assert_array_almost_equal

def test_categorical_loss():
    A = round(5*rand(10,6))
    print A
    loss = categorical_loss(A) 
    lossT = categorical_loss(A, T = True)
    P, V = ones((10,1)), ones((1,loss.n))
   
    loss(P, V)
    lossT(V.T, P.T)
    loss.subgrad(P, V)
    lossT.subgrad(V.T, P.T)
    
    E = loss.encode(A)
    D = loss.decode(E)
    print "A", A
    print "D", D
    assert_array_almost_equal(A, D)

