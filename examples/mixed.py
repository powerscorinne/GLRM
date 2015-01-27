from glrm.loss import QuadraticLoss, HingeLoss, OrdinalLoss
from glrm.reg import QuadraticReg
from glrm import GLRM
from glrm.convergence import Convergence
from glrm.util import pplot, unroll_missing
from numpy.random import randn, choice, seed
from itertools import product
from numpy import sign, ceil, hstack
seed(1)

# Generate problem data
m, k = 50, 10
n1 = 25 # cols of numerical data
n2 = 10 # cols of ordinal data
n3 = 25 # cols of boolean data
n = n1+n2+n3
data = randn(m,k).dot(randn(k,n))
data_real = data[:,:n1] # numerical data
data_ord = data[:,n1:n1+n2] 
data_ord = data_ord - data_ord.min()
data_ord = (data_ord/data_ord.max()*6 + 1).round()# ordinal data, e.g., Likert scale
data_bool = sign(data[:,n1+n2:])

# Initialize model
A = [data_real, data_ord, data_bool]
loss = [QuadraticLoss, OrdinalLoss, HingeLoss]
regX, regY = QuadraticReg(0.01), QuadraticReg(0.01)
converge = Convergence(TOL = 1e-2, max_iters = 1000) # optional (default TOL = 1e-3)
glrm_mix = GLRM(A, loss, regX, regY, k, converge = converge)

# Fit
glrm_mix.fit()

# Results
X, Y = glrm_mix.factors()
A_hat = glrm_mix.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_mix.convergence() # convergence history
pplot([hstack(A), A_hat, hstack(A)-A_hat], ["original", "glrm", "error"])

# Now with missing data
missing = [list(product(range(35, 50), range(n1-5, n1))), list(product(range(35,
    50), range(0, n2))), list(product(range(35, 50), range(0, n3-5)))]

glrm_mix_missing = GLRM(A, loss, regX, regY, k, missing)
glrm_mix_missing.fit()
A_hat = glrm_mix_missing.predict()

# translate missing list into something that we can plot
new_missing = unroll_missing(missing, [n1, n2, n3]) 
pplot([hstack(A), new_missing, A_hat, hstack(A)-A_hat], ["original", "missing", "glrm", "error"])
