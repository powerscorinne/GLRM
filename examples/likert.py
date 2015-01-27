from glrm.loss import OrdinalLoss
from glrm.reg import QuadraticReg
from glrm import GLRM
from glrm.convergence import Convergence
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy import sign
from itertools import product
from math import ceil
seed(1)

# Generate problem data
m, n, k = 500, 500, 20
data = randn(m,k).dot(randn(k,n))
data = data - data.min()
data = (data/data.max()*6).round() + 1 # approx rank k
#data = choice(range(7), (m,n)) + 1 # not inherently rank k

# Initialize model
A = data
loss = OrdinalLoss
regX, regY = QuadraticReg(0.001), QuadraticReg(0.001)
converge = Convergence(TOL = 1e-2, max_iters = 1000) # optional, default TOL = 1e-3
glrm_ord = GLRM(A, loss, regX, regY, k, converge = converge)

# Fit
glrm_ord.fit()

# Results
X, Y = glrm_ord.factors()
A_hat = glrm_ord.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_ord.convergence() # convergence history
pplot([A, A_hat, A-A_hat], ["original", "glrm", "error"])
