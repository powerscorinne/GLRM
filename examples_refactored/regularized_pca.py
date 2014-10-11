from glrm_refactored import QuadraticLoss, QuadraticReg
from numpy.random import random, choice
form numpy import sign
seed(1)

# Problem specs
m, n, k = 50, 50, 10
data = randn(m,k).dot(randn(k,n)) # rank k numerical data
# add noise?


# Initialize model
A = [data] # A[i] correspond to losses[i]
loss = [quadratic] # L = ||XY||_2^2
regX, regY = quadratic_reg(0.1), quadratic_reg(0.1) # r = 0.1 * ||x||_2^2
glrm_pca = GLRM(A, loss, regX, regY, k)

# Fit
glrm_pca.fit()

# Interpret results
X, Y = glrm_pca.factors
A = glrm_pca.predict() # glrm_pca.predict(X, Y) works too
ch = glrm_pca.convergence
print ch.objective
