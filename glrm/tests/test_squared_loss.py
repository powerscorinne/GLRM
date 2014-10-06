from glrm import squared_loss
from numpy import ones, zeros

def test_squared_loss():
    A = ones((4,6))
    A[0,0] = 0
    A[2,2] = 2
    P, V = ones((4,1)), ones((1,6))
    missing = [(0,1)]
    loss = squared_loss(A, False, missing)

    #assert(loss(P, V) == # XXX right answer)
    #assert(loss.subgrad(P, V) == # XXX right answer)
    # encode, decode, n


