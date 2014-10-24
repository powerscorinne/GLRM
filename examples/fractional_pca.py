from glrm.loss import FractionalLoss
from glrm.reg import QuadraticReg
from glrm import GLRM
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy import sign, exp
seed(2)

# Generate problem data
m, n, k = 50, 50, 5
eta = 0.1 # noise power
data = exp(randn(m,k).dot(randn(k,n)) + eta*randn(m,n))+eta*randn(m,n) # noisy rank k

# Initialize model
A = data
loss = FractionalLoss
regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
glrm_frac = GLRM(A, loss, regX, regY, k)

# Fit
glrm_frac.fit()

# Results
X, Y = glrm_frac.factors()
A_hat = glrm_frac.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_frac.convergence() # convergence history
pplot([A, A_hat, A-A_hat], ["original", "glrm", "error"])

# Now with missing data
# from numpy.random import choice
# from itertools import product
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# 
# glrm_pca_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_pca_nn_missing.fit()
# glrm_pca_nn_missing.compare()
