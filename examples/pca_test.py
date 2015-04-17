from numpy import diag, sqrt, tile, hstack, vstack, ones
from numpy.linalg import svd, norm
from numpy.random import randn, seed
from glrm.loss import QuadraticLoss, HuberLoss, HingeLoss
from glrm import GLRM
from glrm.reg import QuadraticReg, LinearReg, NonnegativeReg
seed(1)

def PCA(A, k):
    mean_A = tile(A.mean(0), (A.shape[0],1))
    A0 = A - mean_A

    u, s, v = svd(A0, full_matrices = False)
    u, s, v = u[:,:k], diag(sqrt(s[:k])), v[:k,:]
    X = hstack((u.dot(s), ones((m,1))))
    Y = vstack((s.dot(v), A.mean(0)))

    return X, Y

def GLRMfit(A, k, missing=None):
    loss = QuadraticLoss
    regX, regY = LinearReg(0.001), LinearReg(0.001)
    model = GLRM(A, loss, regX, regY, k, missing)
    model.fit(eps=1e-4, max_iters=1000)
    model.converge.plot()
    return model.factors()

if __name__ == '__main__':
    m, n, k = 100, 50, 10
    A = randn(m,n)
    missing = [[(1,1), (3,5), (10, 10)]]
    X, Y = GLRMfit(A, k, missing)
    Xpca, Ypca = PCA(A, k)

    Z = A-X.dot(Y)
    Zpca = A-Xpca.dot(Ypca)
    for (i,j) in missing[0]: Z[i,j], Zpca[i,j] = 0,0
    print norm(Z)
    print norm(Zpca)
