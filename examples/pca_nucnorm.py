from glrm.loss import QuadraticLoss
from glrm.reg import QuadraticReg
from glrm import GLRM

from numpy.random import randn, choice, seed
from numpy import sign
seed(1)

# Generate problem data
m, n, k = 50, 50, 5
eta = 0.1 # noise power
data = randn(m,k).dot(randn(k,n)) + eta*randn(m,n) # noisy rank k

# Initialize model
A = data
loss = QuadraticLoss
regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
glrm_pca_nn = GLRM(A, loss, regX, regY, k)

# Fit
glrm_pca_nn.fit()

# Results
X, Y = glrm_pca_nn.factors()
A_hat = glrm_pca_nn.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_pca_nn.convergence() # convergence history
glrm_pca_nn.compare() # simple visualization tool to compare A and A_hat

# Now with missing data
# from numpy.random import choice
# from itertools import product
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# 
# glrm_pca_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_pca_nn_missing.fit()
# glrm_pca_nn_missing.compare()
