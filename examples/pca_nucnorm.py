from glrm.loss import QuadraticLoss
from glrm.reg import QuadraticReg, ZeroReg
from glrm import GLRM
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy.random import choice
from itertools import product
from numpy import sign
seed(1)

# Generate problem data
m, n, k = 50, 50, 10
eta = 0.1 # noise power
data = randn(m,k).dot(randn(k,n)) + eta*randn(m,n) # noisy rank k

# Initialize model
A = data
loss = QuadraticLoss
regX, regY = QuadraticReg(0.0001), QuadraticReg(0.0001)
glrm_nn = GLRM(A, loss, regX, regY, k)

# Fit
glrm_nn.fit(eps=1e-4, max_iters=1000)

# Results
X, Y = glrm_nn.factors()
A_hat = glrm_nn.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_nn.convergence() # convergence history
pplot([A, A_hat, A - A_hat], ["original", "glrm", "error"])

# # Now with missing data
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# glrm_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_nn_missing.fit()
# A_hat = glrm_nn_missing.predict()
# pplot([A, missing, A_hat, A - A_hat], \
#         ["original", "missing", "glrm", "error"])
# 
