from glrm import fractional_loss
from numpy.random import rand
from numpy import round, ones
from numpy.testing import assert_array_almost_equal

def test_fractional_loss():
    A = 5*rand(10,6)
    print A
    loss = fractional_loss(A) 
    lossT = fractional_loss(A, T = True)
    print "loss", loss.n
    P, V = rand(10,1), rand(1,loss.n)
   
    loss(P, V)
    lossT(V.T, P.T)
    loss.subgrad(P, V)
    lossT.subgrad(V.T, P.T)
    
