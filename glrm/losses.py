from numpy import unique, maximum, ones, sign, Inf, minimum, round, \
        expand_dims, median, zeros, hstack, vstack, argmax, array, arange, tile, ceil, nan
from numpy.linalg import norm
from utils import min_value, scale
from copy import copy

""" Loss functions for multiple data types """

# XXX add comments for how to write loss, subgrad functions to add new losses,
# and how to use the @scale wrapper

# XXX utlimately, should write a loss meta class, which requires a loss, subgrad
# be defined (and automatically wrap them while instantiating super class) 
# as well as provide default encode/decode, self.n, and __call__ assignments.

class squared_loss(object):
    def __init__(self, A, T = False, missing = []):
        if T: A = A.T # transpose
        
        # V, P are variable, parameter, e.g. X, Y 
        @scale(A, T, missing)
        def loss(P, V): return (A - P.dot(V))**2/2.0
        def subgrad(P, V): return -P.T.dot((A - P.dot(V))*loss.mask) # subgradient w.r.t. B for B = PV
        
        def decode(At): return At
        def encode(At): return At
        
        # save 
        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        self.n = A.shape[1] # number of columns of A
    
    def __str__(self):
        return "squared loss"
    def __call__(self, P, V):
        return self.loss(P, V)

class huber_loss(object):

    # default huber parameter
    a = 1.0

    def __init__(self, A, T = False, missing = []): 
        if T: A = A.T # transpose

        @scale(A, T, missing)
        def loss(P, V): 
            B = A - P.dot(V)
            return ((abs(B) <= self.a)*B**2 + (abs(B) > self.a)*(2*abs(B) -
                self.a)*self.a)

        def subgrad(P, V): 
            B = A - P.dot(V)
            return -P.T.dot((2*(abs(B) <= self.a)*B + (abs(B) >
                self.a)*sign(B)*2*self.a)*loss.mask)
        
        def decode(At): return At
        def encode(At): return At
        
        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        self.n = A.shape[1]

    def __str__(self):
        return "huber loss with parameter {0}".format(self.a)
    def __call__(self, P, V):
        return self.loss(P, V)

class fractional_loss(object):

    def __init__(self, A, T = False, missing = []):
        if T: A = A.T # transpose

        @scale(A, T, missing)
        def loss(P, V):
            U = P.dot(V)
            return maximum((A - U)/U, (U - A)/A)

        def subgrad(P, V):
            U = P.dot(V)
            return -P.T.dot(((1.0/A)*(U >= A) + (-A/U**2)*(U < A))*loss.mask)

        def decode(At): return At
        def encode(At): return At

        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        self.n = A.shape[1]

    def __str__(self):
        return "fractional loss"

    def __call__(self, P, V):
        return self.loss(P, V)


class hinge_loss(object):
    def __init__(self, A, T = False, missing = []):
        # missing entry / scaling filter:
        if T: A = A.T # transpose
        
        @scale(A, T, missing)
        def loss(P, V): return maximum((1 - A*P.dot(V)), 0)
        def subgrad(P, V): return -P.T.dot(A*((1 - A*P.dot(V)) > 0)*loss.mask)
        
        def decode(At): return sign(At)
        def encode(At): return At
        
        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        self.n = A.shape[1]

    def __str__(self):
        return "hinge loss"
    def __call__(self, P, V):
        return self.loss(P, V)


class ordinal_loss(object):
    def __init__(self, A, T = False, missing = [], min_val = None, max_val = None):
        if T: A = A.T # tranpose

        # if not given, find min, max values of A (ignore entries marked as missing)
        if min_val is None: min_val = int(min_value(A, missing))
        if max_val is None: max_val = int(-min_value(-A, missing))

        @scale(A, T, missing)
        def loss(P, V):
            B = P.dot(V)
            C = zeros(B.shape) # cumulative loss for each entry
            for a in range(min_val, max_val):
                C += maximum(B - a, 0)*(a >= A) + maximum(-B + a + 1, 0)*(a < A)
            return C
       
        def subgrad(P, V):
            B = P.dot(V)
            C = (B < min_val)*(min_val - A) + (B > max_val)*(max_val - A) + \
                sign(B - A)*ceil(abs(B-A))*((B >= min_val) & (B <= max_val)) 
            return -P.T.dot(C*loss.mask)
        
        def decode(At): return round(maximum(minimum(At, max_val), min_val))
        # XXX clip?
        def encode(At): return At
        
        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        self.n = A.shape[1]
        self.min_val = min_val
        self.max_val = max_val
    
    def __str__(self):
        return "ordinal loss with compounding error".format(self.min_val, self.max_val)
    def __call__(self, P, V):
        return self.loss(P, V)


class ordinal_loss_lin(object):
    def __init__(self, A, T = False, missing = [], min_val = None, max_val = None):
        if T: A = A.T # tranpose

        # if not given, find min, max values of A (ignore entries marked as missing) 
        if min_val is None: min_val = min_value(A, missing)
        if max_val is None: max_val = -min_value(-A, missing)
        
        @scale(A, T, missing)
        def loss(P, V):
            F = abs(A - P.dot(V))
            return F*(~(((A == min_val) & (P.dot(V) < min_val)) | ((A
                == max_val) & (P.dot(V) > max_val))))
        def subgrad(P, V):
            G =  sign(A - P.dot(V))
            return -P.T.dot(G*(~(((A == min_val) & (P.dot(V) <
                min_val)) | ((A == max_val) & (P.dot(V) > max_val))))*loss.mask)

        def decode(At): return round(maximum(minimum(At, max_val), min_val))
        def encode(At): return At
        
        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        self.n = A.shape[1]
        self.min_val = min_val
        self.max_val = max_val
    
    def __str__(self):
        return "ordinal loss".format(self.min_val, self.max_val)
    def __call__(self, P, V):
        return self.loss(P, V)
    

# ====================== XXX multidimensional loss ===========================

class categorical_loss(object):
    # XXX not complete
    def __init__(self, A0, T = False, missing = []): # XXX add extra categories?
        A = copy(A0)

        # calculate number of unique objects (M) regardless of scalar value
        categories = [unique(A[:,j]) for j in range(A.shape[1])]
        M = [len(cat) for cat in categories] # number of categories per column
        # relabel categories to lie in range(M[j])
        for j in range(A.shape[1]):
            for i in range(len(categories[j])):
                A[A[:,j]==categories[j][i], j] = i
             
        if T: A = A.T # transpose
        
        @scale(A, T, missing, M)
        # XXX clean this up using decode?
        def loss(P, V):
            B = P.dot(V)
            m, n = B.shape
    
            # be careful about which axis we reduce the M choices over, depends on T
            if T: 
                x = (tile(array(M).cumsum() - M, (n,1)).T + A).astype(int)
                y = expand_dims(arange(n),0)
            else: 
                x = expand_dims(arange(m),1)
                y = (tile(array(M).cumsum() - M, (m,1)) + A).astype(int)
            
            # select value of true label
            F = B[x,y]

            # select largest value of non-true label
            B[x,y] = -Inf
            CM = array(M).cumsum()
            if T: G = array([B[(CM[i]-M[i]):CM[i], :].max(0) for i in
                range(len(M))]).reshape(len(M),n)
            else: G = array([B[:,(CM[i]-M[i]):CM[i]].max(1) for i in 
                range(len(M))]).reshape(m,len(M))
            
            # hinge loss of (largest wrong value) - (right value)
            return maximum(1 - F + G, 0)

        def subgrad(P, V):
            B = P.dot(V)
            m, n = B.shape
            C = zeros(B.shape)
            CM = array(M).cumsum()

            if T:
                x = tile(array(M).cumsum() - M, (n,1)).T
                y = expand_dims(arange(n), 0)
                xF = (x + A).astype(int)
                F = B[xF, y]
                B[xF, y] = -Inf
                H = array([B[(CM[i]-M[i]):CM[i], :].argmax(0) for i in
                    range(len(M))]).reshape(len(M),n)
                xG = (x + H).astype(int)
                G = B[xG, y]
                C[xF, y] = -1
                C[xG, y] = 1
                C = ((1 - F + G > 0)*loss.mask).repeat(M, 0)*C
            else: 
                x = expand_dims(arange(m), 1)
                y = tile(array(M).cumsum() - M, (m,1))
                yF = (y + A).astype(int)
                F = B[x, yF]
                B[x, yF] = -Inf
                H = array([B[:,(CM[i]-M[i]):CM[i]].argmax(1) for i in
                    range(len(M))]).reshape(len(M),m).T
                yG = (y + H).astype(int)
                G = B[x, yG]
                C[x, yF] = -1
                C[x, yG] = 1
                C = ((1 - F + G > 0)*loss.mask).repeat(M, 1)*C
           
            return P.T.dot(C)

        # XXX I don't think I need to worry about encoding/decoding for tranpose
        # the encoding/decoding will always be performed on a matrix with
        # features as columns (unless i want to use decode above)
        def encode(A0):
            At = copy(A0)
            # relabel categories to lie in range(M[j])
            for j in range(len(M)):
                for i in range(len(categories[j])):
                    At[At[:,j]==categories[j][i], j] = i
            
            # expand into multidimensional space
            m = At.shape[0]
            Ae = -ones((m, sum(M)))
            indx = tile(array(M).cumsum() - M, (m,1)) + At
            Ae[expand_dims(arange(m), 1), indx.astype(int)] = 1
            return Ae
        
        def decode(At):
            CM = array(M).cumsum()
            # decode each column by picking max and using lookup table
            return hstack([expand_dims(categories[j][At[:,
                CM[j]-M[j]:CM[j]].argmax(1)],1) for j in range(len(M))])

        self.loss = loss
        self.subgrad = subgrad
        self.decode = decode
        self.encode = encode
        if T: self.n = A.shape[1]
        else: self.n = sum(M)
        
    def __str__(self):
        return "categorical (multidimensional) hinge loss"

    def __call__(self, P, V):
        return self.loss(P, V)
