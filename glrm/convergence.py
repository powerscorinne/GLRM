from matplotlib import pyplot as plt

class Convergence(object):

    def __init__(self, TOL = 1e-2, max_iters = 1e3):
        self.obj = []
        self.val = []
        self.TOL = TOL
        self.max_iters = max_iters

    def d(self):
        # return True if converged
        if len(self) < 2: return False
        if len(self) > self.max_iters: 
            print "hit max iters for convergence object"
            return True
        return abs(self.obj[-1] - self.obj[-2])/self.obj[-2] < self.TOL

    def __len__(self):
        return len(self.obj)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return str(self.obj)

    def plot(self):
        plt.plot(self.obj)
        plt.title("model error")
        plt.xlabel("iteration")
        plt.show()
