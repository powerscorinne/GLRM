from glrm.loss import HuberLoss
from glrm.reg import QuadraticReg
from glrm import GLRM
from glrm.util import pplot
from numpy.random import randn, choice, seed
from numpy import sign
from random import sample
from math import sqrt
from itertools import product
from matplotlib import pyplot as plt
seed(1)

# Generate problem data
m, n, k = 50, 50, 5
sym_noise = 0.2*sqrt(k)*randn(m,n)
asym_noise = sqrt(k)*randn(m,n) + 3*abs(sqrt(k)*randn(m,n)) # large, sparse noise
rate = 0.3 # percent of entries that are corrupted by large, outlier noise
corrupted_entries = sample(list(product(range(m), range(n))), int(m*n*rate))
data = randn(m,k).dot(randn(k,n))
A = data + sym_noise
for ij in corrupted_entries: A[ij] += asym_noise[ij]

# Initialize model
loss = HuberLoss
regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
glrm_pca_nn = GLRM(A, loss, regX, regY, k)

# Fit
glrm_pca_nn.fit()

# Results
X, Y = glrm_pca_nn.factors()
A_hat = glrm_pca_nn.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_pca_nn.convergence() # convergence history
pplot([data, A, A_hat, data-A_hat], ["original", "corrupted", "glrm", "error"])


# Now with missing data
# from numpy.random import choice
# from itertools import product
# missing = list(product(range(int(0.25*m), int(0.75*m)), range(int(0.25*n), int(0.75*n))))
# 
# glrm_pca_nn_missing = GLRM(A, loss, regX, regY, k, missing)
# glrm_pca_nn_missing.fit()
# glrm_pca_nn_missing.compare()
