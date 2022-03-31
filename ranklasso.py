import numpy as np
import numpy.random as rgt
import math
from scipy.optimize import linprog

class ranklassoregression():


    def __init__(self, X, Y, intercept=True):

        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.

        Y : n-dimensional vector of response variables.

        intercept : logical flag for adding an intercept to the model.

        options : a dictionary of internal statistical and optimization parameters.

            phi : initial quadratic coefficient parameter in the ILAMM algorithm; default is 0.1.

            gamma : adaptive search parameter that is larger than 1; default is 1.25.

            max_iter : maximum numder of iterations in the ILAMM algorithm; default is 1e3.

            tol : the ILAMM iteration stops when |beta^{k+1} - beta^k|^2/|beta^k|^2 <= tol; default is 1e-5.

            irw_tol : tolerance parameter for stopping iteratively reweighted L1-penalization; default is 1e-4.

            nsim : number of simulations for computing a data-driven lambda; default is 200.

        '''
        self.n, self.p = X.shape
        self.Y = Y.reshape(self.n)
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.X1 = np.c_[np.ones(self.n, ), (X - self.mX) / self.sdX]
        else:
            self.X, self.X1 = X, X / self.sdX

    def ranklasso(self, Lambda = np.array([]), numsim=500, c=1.01, alpha0=0.1,method='highs'):
        X = self.X
        Y = self.Y
        n = self.n
        p = self.p
        'Simulated choice of the penalty parameter'
        if not np.array(Lambda).any():
            G=np.zeros(numsim)
            for i in range(numsim):
                G[i] = np.max(-2*X.T.dot(2*rgt.permutation(np.arange(self.n)+1)-(self.n+1))/(self.n*(self.n-1)))
            Lambda = c*np.quantile(G, 1-alpha0)
        else:
            Lambda=Lambda


        a = np.zeros(n ** 2)
        for i in range(n):
            for j in range(n):
                a[n * i + j] = 1
            a[n * i + i] = 0

        c = np.zeros(2 * n ** 2 + 2 * p)
        c[0:2 * n ** 2] = np.append(a, a) / (n * (n - 1))
        c[2 * n ** 2:] = np.append(Lambda * np.ones(p), np.zeros(p))

        Aub = np.zeros((2 * n ** 2 + 2 * p, 2 * n ** 2 + 2 * p))
        for i in range(2 * n ** 2):
            Aub[i, i] = -1
        for j in range(p):
            Aub[2 * n ** 2 + j, 2 * n ** 2 + j] = -1
            Aub[2 * n ** 2 + j, 2 * n ** 2 + j + p] = 1
            Aub[2 * n ** 2 + j + p, 2 * n ** 2 + j] = -1
            Aub[2 * n ** 2 + j + p, 2 * n ** 2 + j + p] = -1
        Aeq = np.zeros((n ** 2, 2 * n ** 2 + 2 * p))
        beq = np.zeros(n ** 2)
        for i in range(n):
            for j in range(n):
                Aeq[n * i + j, n * i + j] = 1
                Aeq[n * i + j, n ** 2 + n * i + j] = -1
                Aeq[n * i + j, 2 * n ** 2 + p:] = X[i] - X[j]
                beq[n * i + j] = Y[i] - Y[j]
        res = linprog(c, A_ub=Aub, b_ub=np.zeros(2 * n ** 2 + 2 * p), A_eq=Aeq, b_eq=beq, bounds=[None, None],method=method)
        return {'beta': res['x'][2*n**2+p:],  'lambda': Lambda}


