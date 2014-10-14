from glrm.loss import OrdinalLoss
from glrm.reg import QuadraticReg
from glrm import GLRM
from glrm.convergence import Convergence
from numpy.random import randn, choice, seed
from numpy import sign
from itertools import product
from math import ceil
seed(1)

# Generate problem data
m, n, k = 50, 50, 20
data = randn(m,k).dot(randn(k,n))
data = ((data - data.min())/data.max()*6).round() + 1 # approx rank k
#data = choice(range(7), (m,n)) + 1 # not inherently rank k

# Initialize model
A = data
converge = Convergence(TOL = 1e-4, max_iters = 1000)
loss = OrdinalLoss
regX, regY = QuadraticReg(0.01), QuadraticReg(0.01) # r = 0.1 * ||x||_2^2
glrm_ord = GLRM(A, loss, regX, regY, k, converge = converge)

# Fit
glrm_ord.fit()

# Results
X, Y = glrm_ord.factors()
A_hat = glrm_ord.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_ord.convergence() # convergence history
glrm_ord.compare() # simple visualization tool to compare A and A_hat

# Now with missing data
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# 
# glrm_pca_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_pca_nn_missing.fit()
# glrm_pca_nn_missing.compare()
