from functions import ordinal_loss_flat
from numpy.random import rand, seed, randn
from numpy import ones, zeros, round, hstack

M = 4
m, n, k = 6, 3, 2

seed(1)

A = round((M-1)*rand(m,n))
X, Y = randn(m,k), randn(k+1,n)
X = hstack((X, ones((m,1))))

loss = ordinal_loss_flat(A)


