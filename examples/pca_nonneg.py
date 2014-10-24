from glrm.loss import QuadraticLoss
from glrm.reg import NonnegativeReg, QuadraticReg
from glrm import GLRM
from glrm.util import pplot
from glrm.algs import ProxGD
from numpy.random import randn, choice, seed
from numpy.random import choice
from itertools import product
from numpy import sign

# Generate problem data
m, n, k = 20, 20, 5
eta = 0.1 # noise power
X_true, Y_true = abs(randn(m,k)), abs(randn(k,n))
data = X_true.dot(Y_true) + eta*randn(m,n) # noisy rank k

# Initialize model
A = data
loss = QuadraticLoss
regX, regY = NonnegativeReg(0.1), NonnegativeReg(0.1)
glrm_nn = GLRM(A, loss, regX, regY, k, algX = ProxGD, algY = ProxGD)

# Fit
glrm_nn.fit()

# Results
X, Y = glrm_nn.factors()
A_hat = glrm_nn.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_nn.convergence() # convergence history
pplot([A, A_hat, A - A_hat], ["original", "glrm", "error"])

# Now with missing data
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# glrm_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_nn_missing.fit()
# A_hat = glrm_nn_missing.predict()
# pplot([A, missing, A_hat, A - A_hat], \
#         ["original", "missing", "glrm", "error"])
# 
