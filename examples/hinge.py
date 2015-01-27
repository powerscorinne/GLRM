from glrm.loss import HingeLoss
from glrm.reg import NonnegativeReg, QuadraticReg
from glrm import GLRM
from glrm.util import pplot
from glrm.convergence import Convergence
from numpy.random import randn, choice, seed
from numpy.random import choice
from itertools import product
from numpy import sign

# Generate problem data
m, n, k = 1000, 1000, 20
eta = 0.1 # noise power
X_true, Y_true = randn(m,k), randn(k,n)
data = sign(X_true.dot(Y_true) + eta*randn(m,n)) # noisy rank k

# Initialize model
A = data
loss = HingeLoss
regX, regY = QuadraticReg(0.5), QuadraticReg(0.5)
c = Convergence(TOL=1e-6)
model = GLRM(A, loss, regX, regY, k, converge=c)

# Fit
model.fit(eps=1e-1, max_iters=20)

# Results
X, Y = model.factors()
A_hat = model.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = model.convergence() # convergence history
pplot([A, A_hat, A - A_hat], ["original", "glrm", "error"])
# 
# # Now with missing data
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# glrm_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_nn_missing.fit()
# A_hat = glrm_nn_missing.predict()
# pplot([A, missing, A_hat, A - A_hat], \
#         ["original", "missing", "glrm", "error"])
# 
