from paths import *
from glrm import GLRM, categorical_loss, hinge_loss, norm2sq_reg, squared_loss
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, diag
from numpy.linalg import norm, inv, svd
from glrm.utils.pretty_plot import visualize_recovery
from time import time
from math import sqrt
seed(1)

glrm_rms = []
pca_rms = []
glrm_mce = []
pca_mce = []

for i in range(10):
    ## =================== Problem data ==============================
    m, n, k = 50, 50, 10
    A = sign(randn(m,k).dot(randn(k,n)))
    As = [A]

    ## ================== GLRM ======================
    losses = [categorical_loss]
    regsY, regX = norm2sq_reg(0.1), norm2sq_reg(0.1) # parameter is lambda_i
    model = GLRM(As, losses, regsY, regX, k, svd_init=False)
    At = model.alt_min(outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters = 
            100, inner_max_iters = 1000, quiet = True)
    At = At[0]

    #visualize_recovery(A, At, "boolean data", "glrm rank {0} recovery".format(k), n, "boolean_glrm")
    mce = float((A != At).sum())/(m*n)
    rms = sqrt(float(((A != At)**2).sum())/(m*n))

    glrm_mce.append(mce)
    glrm_rms.append(rms)
    #print "percent of misclassified boolean data: "
    #print "GLRM: {0:.2f}% misclassified points".format(100*mce)


    ## =================== regularized PCA =====================
    losses = [squared_loss]
    model = GLRM(As, losses, regsY, regX, k, svd_init=False)
    At = model.alt_min(outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters = 
            100, inner_max_iters = 1000, quiet = True)
    At = sign(At[0])

    #visualize_recovery(A, At, "boolean data", "pca rank {0} recovery".format(k), n, "boolean_pca")
    mce = float((A != At).sum())/(m*n)
    rms = sqrt(float(((A != At)**2).sum())/(m*n))
    pca_mce.append(mce)
    pca_rms.append(rms)
    #print "PCA: {0:.2f}% misclassified points".format(100*mce)

from numpy import mean
from math import sqrt
print sqrt(sum((glrm_mce - mean(glrm_mce))**2)/len(glrm_mce))
print sqrt(sum((pca_mce - mean(pca_mce))**2)/len(pca_mce))
print mean(glrm_rms)
print mean(pca_rms)

