from glrm.loss import HingeLoss
from glrm.reg import QuadraticReg
from glrm import GLRM
from glrm.convergence import Convergence
from numpy.random import randn, choice, seed
from numpy import sign, ones
from itertools import product
seed(1)

# Generate problem data (draw smiley with -1's, 1's)
m, n, k = 500, 500, 8
data = -ones((m, n))
for i,j in product(range(120, 190), range(120, 190)): 
    d = (155-i)**2 + (155-j)**2
    if d < 35**2: 
        data[i,j] = 1
        data[i, m-j] = 1
for i,j in product(range(300, 450), range(100, 250)):
    d = (250 - i)**2 + (250-j)**2
    if d < 200**2 and d > 150**2: 
        data[i,j] = 1
        data[i,m-j] = 1

# Initialize model
A = data
loss = HingeLoss
regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)
converge = Convergence(TOL = 1e-2)
glrm_binary = GLRM(A, loss, regX, regY, k, converge = converge)

# Fit
glrm_binary.fit()

# Results
X, Y = glrm_binary.factors()
A_hat = glrm_binary.predict() # glrm_pca.predict(X, Y) works too; returns decode(XY)
ch = glrm_binary.convergence() # convergence history
glrm_binary.compare() # simple visualization tool to compare A and A_hat

# Now with missing data
missing = list(product(range(int(0.3*m), int(0.8*m)), range(int(0.55*n), int(0.7*n))))

glrm_binary_missing = GLRM(A, loss, regX, regY, k, missing)
glrm_binary_missing.fit()
glrm_binary_missing.compare()
