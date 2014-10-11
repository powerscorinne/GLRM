from numpy import ones, sign, round
from loss import *
from numpy.random import randn, choice

A = choice(range(1,8), 25).reshape(5,5)
X = ones((5,2))
Y = ones((2,5))
q = OrdinalLoss(A)
print q(X, Y)
print q.subgrad(X, Y)
