from numpy import zeros, maximum, round, ones, ceil, sign
from numpy.random import rand
from functions import ordinal_loss

M = 5
A = round((M-1)*rand(6,3))
B = ones((6,3))
missing = [(1, 1), (2, 2), (3, 0)]

loss = ordinal_loss(A, missing = missing)

P = ones((6,2))
V = -ones((2,3))/2.0


