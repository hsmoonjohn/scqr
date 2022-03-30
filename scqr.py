import numpy as np
import numpy.random as rgt
from scipy.stats import norm
import math
from scipy.optimize import linprog

class high_dim(low_dim):
    '''
        Regularized Convolution Smoothed Composite Quantile Regression via ILAMM
                        (iterative local adaptive majorize-minimization)
    '''
    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    weights = ['Multinomial', 'Exponential', 'Rademacher']
    penalties = ["L1", "SCAD", "MCP", "CapppedL1"]
    opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-5,
           'irw_tol': 1e-5, 'nsim': 200}

    def __init__(self, X, Y, intercept=True, options={}):

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

        self.opt.update(options)

    def smooth_check(self, x, tau, h=None, kernel='Laplacian', w=np.array([])):
        if h == None: h = self.bandwidth(tau)

        loss1 = lambda x: np.where(x >= 0, tau * x, (tau - 1) * x) + 0.5 * h * np.exp(-abs(x) / h)
        loss2 = lambda x: (tau - norm.cdf(-x / h)) * x + 0.5 * h * np.sqrt(2 / np.pi) * np.exp(-(x / h) ** 2 / 2)
        loss3 = lambda x: tau * x + h * np.log(1 + np.exp(-x / h))
        loss4 = lambda x: (tau - 0.5) * x + h * (0.25 * (x / h) ** 2 + 0.25) * (abs(x) < h) \
                          + 0.5 * abs(x) * (abs(x) >= h)
        loss5 = lambda x: (tau - 0.5) * x + 0.5 * h * (0.75 * (x / h) ** 2 \
                                                       - (x / h) ** 4 / 8 + 3 / 8) * (abs(x) < h) + 0.5 * abs(x) * (
                                  abs(x) >= h)
        loss_dict = {'Laplacian': loss1, 'Gaussian': loss2, 'Logistic': loss3, \
                     'Uniform': loss4, 'Epanechnikov': loss5}
        if not w.any():
            return np.mean(loss_dict[kernel](x))
        else:
            return np.mean(loss_dict[kernel](x) * w)

    def cqr_smooth_check(self, x, alpha=np.array([]), tau=np.array([]), h=None, kernel='Laplacian', w=np.array([])):
        cqrsc = np.zeros(len(tau))
        for i in range(0, len(tau)):
            cqrsc[i] = self.smooth_check(x - alpha[i], tau[i], h, kernel, w)
        return np.mean(cqrsc)

    def cqr_check_sum(self, x, tau, alpha):
        ccs = 0
        for i in range(0, len(tau)):
            ccs = ccs + np.sum(np.where(x - alpha[i] >= 0, tau[i] * (x - alpha[i]), (tau[i] - 1) * (x - alpha[i])))

        return ccs / len(tau)

    def cqrprox(self, v, a, tau):
        return v-np.maximum((tau-1)/a, np.minimum(v, tau/a))

    def conquer_weight(self, x, tau, kernel="Laplacian", w=np.array([])):
        ker1 = lambda x: 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))
        ker2 = lambda x: norm.cdf(x)
        ker3 = lambda x: 1 / (1 + np.exp(-x))
        ker4 = lambda x: np.where(x > 1, 1, 0) + np.where(abs(x) <= 1, 0.5 * (1 + x), 0)
        ker5 = lambda x: 0.25 * (2 + 3 * x / 5 ** 0.5 - (x / 5 ** 0.5) ** 3) * (abs(x) <= 5 ** 0.5) + (x > 5 ** 0.5)
        ker_dict = {'Laplacian': ker1, 'Gaussian': ker2, 'Logistic': ker3, \
                    'Uniform': ker4, 'Epanechnikov': ker5}
        if not w.any():
            return (ker_dict[kernel](x) - tau) / len(x)
        else:
            return w * (ker_dict[kernel](x) - tau) / len(x)

    def cqr_conquer_weight(self, x, alpha=np.array([]), tau=np.array([]), h=None, kernel="Laplacian", w=np.array([])):
        cqr_cw = self.conquer_weight((alpha[0] - x) / h, tau[0], kernel, w)
        for i in range(1, len(tau)):
            cqr_cw = np.hstack((cqr_cw, self.conquer_weight((alpha[i] - x) / h, tau[i], kernel, w)))
        return cqr_cw / len(tau)

    def cqr_conquer_lambdasim(self, tau=np.array([])):
        cqr_lambda = (rgt.uniform(0, 1, self.n) <= tau[0]) - tau[0]
        for i in range(1, len(tau)):
            cqr_lambda = np.hstack((cqr_lambda, (rgt.uniform(0, 1, self.n) <= tau[i]) - tau[i]))
        return cqr_lambda / (len(tau) * self.n)

    def cqr_self_tuning(self, XX, tau=np.array([])):
        cqr_lambda_sim = np.array([max(abs(XX.dot(self.cqr_conquer_lambdasim(tau))))
                                   for b in range(self.opt['nsim'])])
        return 2*cqr_lambda_sim


    def bandwidth(self, tau):
        h0 = (np.log(self.p) / self.n) ** 0.25
        return max(0.05, h0 * (tau - tau ** 2) ** 0.5)

    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x) * np.where(tmp <= 0, 0, tmp)

    def self_tuning(self, tau=0.5, standardize=True):
        '''
            A Simulation-based Approach for Choosing the Penalty Level (Lambda)

        Reference
        ---------
        l1-Penalized Quantile Regression in High-dimensinoal Sparse Models (2011)
        by Alexandre Belloni and Victor Chernozhukov
        The Annals of Statistics 39(1): 82--130.

        Arguments
        ---------
        tau : quantile level; default is 0.5.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        Returns
        -------
        lambda_sim : a numpy array of self.opt['nsim'] simulated lambda values.
        '''
        if standardize:
            X = self.X1
        else:
            X = self.X
        lambda_sim = np.array([max(abs(X.T.dot(tau - (rgt.uniform(0, 1, self.n) <= tau))))
                               for b in range(self.opt['nsim'])])
        return lambda_sim / self.n

    def concave_weight(self, x, penalty="SCAD", a=None):
        if penalty == "SCAD":
            if a == None: a = 3.7
            tmp = 1 - (abs(x) - 1) / (a - 1)
            tmp = np.where(tmp <= 0, 0, tmp)
            return np.where(tmp > 1, 1, tmp)
        elif penalty == "MCP":
            if a == None: a = 3
            tmp = 1 - abs(x) / a
            return np.where(tmp <= 0, 0, tmp)
        elif penalty == "CapppedL1":
            if a == None: a = 3
            return abs(x) <= a / 2

    def cqrp_admm(self, Lambda=np.array([]), tau=np.array([]), sg=0.01, alpha0=np.array([]),
               beta0=np.array([]), e1=1e-3, e2=1e-3,maxit=20000, lambdaparameter=1.3):
        p = self.p
        n = self.n
        K = len(tau)
        if not beta0.any():
            beta0 = np.zeros(p)
        if not alpha0.any():
            alpha0 = np.zeros(K)
        count = 0

        XX = np.tile(self.X, (K, 1))
        XXX = np.tile(self.X.T, K)

        if not np.array(Lambda).any():
            Lambda = lambdaparameter * np.quantile(self.cqr_self_tuning(XXX, tau), 0.95)


        bmatrix = np.zeros((n * K, K))
        for i in range(0, K):
            bmatrix[n * i:n * (i + 1), i] = 1
        X1 = np.hstack((bmatrix,XX))
        X2 = np.hstack((np.zeros((p, K)), np.identity(p)))
        m = X1.T.dot(X1)+X2.T.dot(X2)
        im = np.linalg.inv(m)
        y = np.tile(self.Y, K)
        phi0 = np.concatenate((alpha0,beta0))
        z0 = y-X1.dot(phi0)
        gamma0 = beta0
        u0 = np.zeros(n*K)
        v0 = np.zeros(p)
        betaseq=np.zeros((p,maxit))

        while count < maxit:
            'update phi'
            phi1 = (1/sg)*np.matmul(im, X1.T.dot(sg*(y-z0)-u0)+X2.T.dot(sg*gamma0+v0))

            'update z and gamma'
            z1 = self.cqrprox(y-X1.dot(phi1)-u0/sg, n*K*sg, np.kron(tau,np.ones(n)))
            gamma1 = self.soft_thresh(phi1[K:]-v0/sg,Lambda/sg)

            'update u and v'
            u1 = u0+sg*(z1+X1.dot(phi1)-y)
            v1 = v0+sg*(gamma1-X2.dot(phi1))

            'check stopping criteria'
            c1 = np.linalg.norm(np.vstack((X1,-X2)).dot(phi1)+np.concatenate((z1,gamma1))-np.concatenate((y,np.zeros(p))))
            c2 = max(np.linalg.norm(np.vstack((X1, -X2)).dot(phi1)), np.linalg.norm(np.concatenate((z1, gamma1))), np.linalg.norm(y))
            c3 = sg*np.linalg.norm(X1.T.dot(z1-z0)-X2.T.dot(gamma1-gamma0))
            c4 = np.linalg.norm(X1.T.dot(u1)-X2.T.dot(v1))
            betaseq[:, count] = gamma1
            if c1 <= e1*math.sqrt(n*K+p)+e2*c2 and c3 <= e1*math.sqrt(n*K+p)+e2*c4:
                count = maxit
            else:
                count = count + 1
            phi0, z0, gamma0, u0, v0 = phi1, z1, gamma1, u1, v1


        return {'alpha': phi0[:K], 'beta': gamma0, 'lambda': Lambda, 'z':z0, 'u':u0, 'v':v0,'betaseq':betaseq}

    def cqrp_admm_smw(self, Lambda=np.array([]), tau=np.array([]), sg=0.01, alpha0=np.array([]),
               beta0=np.array([]), e1=1e-3, e2=1e-3,maxit=20000, lambdaparameter=0.97):
        p = self.p
        n = self.n
        K = len(tau)
        if not beta0.any():
            beta0 = np.zeros(p)
        if not alpha0.any():
            alpha0 = np.zeros(K)
        count = 0

        XX = np.tile(self.X, (K, 1))
        XXX = np.tile(self.X.T, K)

        if not np.array(Lambda).any():
            Lambda = lambdaparameter * np.quantile(self.cqr_self_tuning(XXX, tau), 0.95)


        bmatrix = np.zeros((n * K, K))
        for i in range(0, K):
            bmatrix[n * i:n * (i + 1), i] = 1
        X1 = np.hstack((bmatrix,XX))
        X2 = np.hstack((np.zeros((p, K)), np.identity(p)))

        X0 = self.X - self.X.mean(axis=0)

        Sinv=np.eye(p)-K*X0.T.dot(np.linalg.inv(np.eye(n)+K*X0.dot(X0.T))).dot(X0)
        im1 = np.hstack(((1/n)*np.eye(K),np.zeros((K,p))))
        im2 = np.hstack((np.zeros((p,K)),Sinv))
        im = np.vstack((im1,im2))
        y = np.tile(self.Y, K)
        phi0 = np.concatenate((alpha0,beta0))
        z0 = y-X1.dot(phi0)
        gamma0 = beta0
        u0 = np.zeros(n*K)
        v0 = np.zeros(p)
        betaseq=np.zeros((p,maxit))

        while count < maxit:
            'update phi'
            phi1 = (1/sg)*np.matmul(im, X1.T.dot(sg*(y-z0)-u0)+X2.T.dot(sg*gamma0+v0))

            'update z and gamma'
            z1 = self.cqrprox(y-X1.dot(phi1)-u0/sg, n*K*sg, np.kron(tau,np.ones(n)))
            gamma1 = self.soft_thresh(phi1[K:]-v0/sg,Lambda/sg)

            'update u and v'
            u1 = u0+sg*(z1+X1.dot(phi1)-y)
            v1 = v0+sg*(gamma1-X2.dot(phi1))

            'check stopping criteria'
            c1 = np.linalg.norm(np.vstack((X1,-X2)).dot(phi1)+np.concatenate((z1,gamma1))-np.concatenate((y,np.zeros(p))))
            c2 = max(np.linalg.norm(np.vstack((X1, -X2)).dot(phi1)), np.linalg.norm(np.concatenate((z1, gamma1))), np.linalg.norm(y))
            c3 = sg*np.linalg.norm(X1.T.dot(z1-z0)-X2.T.dot(gamma1-gamma0))
            c4 = np.linalg.norm(X1.T.dot(u1)-X2.T.dot(v1))
            betaseq[:, count] = gamma1
            if c1 <= e1*math.sqrt(n*K+p)+e2*c2 and c3 <= e1*math.sqrt(n*K+p)+e2*c4:
                count = maxit
            else:
                count = count + 1
            phi0, z0, gamma0, u0, v0 = phi1, z1, gamma1, u1, v1


        return {'alpha': phi0[:K], 'beta': gamma0, 'lambda': Lambda, 'z':z0, 'u':u0, 'v':v0,'betaseq':betaseq}

    def cqrp_admm_smw_withoutdemean(self, Lambda=np.array([]), tau=np.array([]), sg=0.01, alpha0=np.array([]),
               beta0=np.array([]), e1=1e-3, e2=1e-3,maxit=20000, lambdaparameter=1.32):
        p = self.p
        n = self.n
        K = len(tau)
        if not beta0.any():
            beta0 = np.zeros(p)
        if not alpha0.any():
            alpha0 = np.zeros(K)
        count = 0

        XX = np.tile(self.X, (K, 1))
        XXX = np.tile(self.X.T, K)

        if not np.array(Lambda).any():
            Lambda = lambdaparameter * np.quantile(self.cqr_self_tuning(XXX, tau), 0.5)


        bmatrix = np.zeros((n * K, K))
        for i in range(0, K):
            bmatrix[n * i:n * (i + 1), i] = 1
        X1 = np.hstack((bmatrix,XX))
        X2 = np.hstack((np.zeros((p, K)), np.identity(p)))

        X0 = self.X - self.X.mean(axis=0)

        Sinv=np.eye(p)-K*X0.T.dot(np.linalg.inv(np.eye(n)+K*X0.dot(X0.T))).dot(X0)
        im1 = np.hstack(((1/n)*np.eye(K)+(1/n**2)*np.ones((K, n)).dot(self.X).dot(Sinv).dot(self.X.T).dot(np.ones((n, K))), -(1/n)*np.ones((K, n)).dot(self.X).dot(Sinv)))
        im2 = np.hstack((-(1/n)*Sinv.dot(self.X.T).dot(np.zeros((n, K))),Sinv))
        im = np.vstack((im1,im2))
        y = np.tile(self.Y, K)
        phi0 = np.concatenate((alpha0,beta0))
        z0 = y-X1.dot(phi0)
        gamma0 = beta0
        u0 = np.zeros(n*K)
        v0 = np.zeros(p)
        betaseq=np.zeros((p,maxit))

        while count < maxit:
            'update phi'
            phi1 = (1/sg)*np.matmul(im, X1.T.dot(sg*(y-z0)-u0)+X2.T.dot(sg*gamma0+v0))

            'update z and gamma'
            z1 = self.cqrprox(y-X1.dot(phi1)-u0/sg, n*K*sg, np.kron(tau,np.ones(n)))
            gamma1 = self.soft_thresh(phi1[K:]-v0/sg,Lambda/sg)

            'update u and v'
            u1 = u0+sg*(z1+X1.dot(phi1)-y)
            v1 = v0+sg*(gamma1-X2.dot(phi1))

            'check stopping criteria'
            c1 = np.linalg.norm(np.vstack((X1,-X2)).dot(phi1)+np.concatenate((z1,gamma1))-np.concatenate((y,np.zeros(p))))
            c2 = max(np.linalg.norm(np.vstack((X1, -X2)).dot(phi1)), np.linalg.norm(np.concatenate((z1, gamma1))), np.linalg.norm(y))
            c3 = sg*np.linalg.norm(X1.T.dot(z1-z0)-X2.T.dot(gamma1-gamma0))
            c4 = np.linalg.norm(X1.T.dot(u1)-X2.T.dot(v1))
            betaseq[:, count] = gamma1
            if c1 <= e1*math.sqrt(n*K+p)+e2*c2 and c3 <= e1*math.sqrt(n*K+p)+e2*c4:
                count = maxit
            else:
                count = count + 1
            phi0, z0, gamma0, u0, v0 = phi1, z1, gamma1, u1, v1


        return {'alpha': phi0[:K], 'beta': gamma0, 'lambda': Lambda, 'z':z0, 'u':u0, 'v':v0,'betaseq':betaseq}

    def cqr_l1(self, Lambda=np.array([]), tau=np.array([]), h=None, kernel="Laplacian", alpha0=np.array([]),
               beta0=np.array([]), res=np.array([]), standardize=True, adjust=True, weight=np.array([]),lambdaparameter=0.97):
        '''
            L1-Penalized Convolution Smoothed Composite Quantile Regression (l1-cqr-conquer)

        Arguments
        ---------
        Lambda : regularization parameter. This should be either a scalar, or
                 a vector of length equal to the column dimension of X. If unspecified,
                 it will be computed by self.self_tuning().

        tau : a vector of quantile levels; has K elements.

        h : bandwidth/smoothing parameter. The default is computed by self.bandwidth().

        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        beta0 : initial estimator. If unspecified, it will be set as zero.

        alpha0 : initial estimator for intercept terms in CQR regression (alpha terms). If unspecified, it will be set as zero.

        res : residual vector of the initial estiamtor.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        adjust : logical flag for returning coefficients on the original scale.

        weight : an n-vector of observation weights; default is np.array([]) (empty).

        Returns
        -------
        'alpha': a numpy array of estimated coefficients for alpha terms.
        'beta' : a numpy array of estimated coefficients for beta terms.

        'res' : an n-vector of fitted residuals.

        'niter' : number of iterations.

        'lambda' : lambda value.

        '''
        K = len(tau)

        if standardize:
            X = self.X1
        else:
            X = self.X

        XX = np.tile(X.T, K)

        if not np.array(Lambda).any():
            Lambda = lambdaparameter * np.quantile(self.cqr_self_tuning(XX, tau), 0.95)
        if h == None: h = self.bandwidth(max(tau))

        if not beta0.any():
            '''
            model = self.l1_retire(Lambda, np.mean(tau), standardize=standardize, adjust=False)
            beta0, res, count = model['beta'], model['res'], model['niter']
            '''
            beta0 = np.zeros(self.p)
            count = 0
        else:
            count = 0

        if not alpha0.any():
            alpha0 = np.zeros(K)

        if not res.any():
            res = self.Y - X.dot(beta0)

        alphaX = np.zeros((K, self.n * K))
        for i in range(0, K):
            for j in range(i * self.n, (i + 1) * self.n):
                alphaX[i, j] = 1

        phi, r0 = self.opt['phi'], 1
        while r0 > self.opt['tol'] * (np.sum(beta0 ** 2) + np.sum(alpha0 ** 2)) and count < self.opt['max_iter']:

            gradalpha0 = alphaX.dot(self.cqr_conquer_weight(res, alpha0, tau, h, kernel, w=weight))
            gradbeta0 = XX.dot(self.cqr_conquer_weight(res, alpha0, tau, h, kernel, w=weight))
            loss_eval0 = self.cqr_smooth_check(res, alpha0, tau, h, kernel, weight)
            alpha1 = alpha0 - gradalpha0 / phi
            beta1 = beta0 - gradbeta0 / phi
            beta1 = self.soft_thresh(beta1, Lambda / phi)
            diff_alpha = alpha1 - alpha0
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta) + diff_alpha.dot(diff_alpha)
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(gradbeta0) + diff_alpha.dot(gradalpha0) + 0.5 * phi * r0
            loss_eval1 = self.cqr_smooth_check(res, alpha1, tau, h, kernel, weight)

            while loss_proxy < loss_eval1:
                phi *= self.opt['gamma']
                alpha1 = alpha0 - gradalpha0 / phi
                beta1 = beta0 - gradbeta0 / phi
                beta1 = self.soft_thresh(beta1, Lambda / phi)
                diff_alpha = alpha1 - alpha0
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta) + diff_alpha.dot(diff_alpha)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(gradbeta0) + diff_alpha.dot(gradalpha0) + 0.5 * phi * r0
                loss_eval1 = self.cqr_smooth_check(res, alpha1, tau, h, kernel, weight)

            alpha0, beta0, phi = alpha1, beta1, self.opt['phi']
            count += 1

        if standardize and adjust:
            beta1 = beta1 / self.sdX

        return {'alpha': alpha1, 'beta': beta1, 'res': res, 'niter': count, 'lambda': Lambda, 'h': h}

    def cqr_irw(self, Lambda=None, tau=np.array([]), h=None, kernel="Laplacian", alpha0=np.array([]),
                beta0=np.array([]), res=np.array([]),
                penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True, weight=np.array([]),lambdaparameter=1.6):
        '''
            Iteratively Reweighted L1-Penalized Conquer (irw-l1-conquer)

        Arguments
        ----------
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".

        a : the constant (>2) in the concave penality; default is 3.7.

        nstep : number of iterations/steps of the IRW algorithm; default is 5.

        Returns
        -------
        'beta' : a numpy array of estimated coefficients.

        'res' : an n-vector of fitted residuals.

        'nirw' : number of reweighted penalization steps.

        'lambda' : lambda value.
        '''

        K = len(tau)
        if standardize:
            X = self.X1
        else:
            X = self.X

        XX = np.tile(X.T, K)

        if Lambda == None:
            Lambda = lambdaparameter * np.quantile(self.cqr_self_tuning(XX, tau), 0.95)
        if h == None: h = self.bandwidth(max(tau))

        if not beta0.any():
            model = self.cqr_l1(Lambda, tau, h, kernel, alpha0=np.zeros(K), beta0=np.zeros(self.p),
                                standardize=standardize, adjust=False, weight=weight)
        else:
            model = self.cqr_l1(Lambda, tau, h, kernel=kernel, alpha0=alpha0, beta0=beta0, res=res,
                                standardize=standardize, adjust=False, weight=weight)
        alpha0, beta0, res = model['alpha'], model['beta'], model['res']

        err, count = 1, 1
        while err > self.opt['irw_tol'] and count <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0 / Lambda, penalty, a)
            model = self.cqr_l1(rw_lambda, tau, h, kernel, alpha0, beta0, res, standardize, adjust=False, weight=weight)
            err = (np.sum((model['beta'] - beta0) ** 2) + np.sum((model['alpha'] - alpha0) ** 2)) / (
                        np.sum(beta0 ** 2) + np.sum(alpha0 ** 2))
            alpha0, beta0, res = model['alpha'], model['beta'], model['res']
            count += 1

        if standardize and adjust:
            beta0 = beta0 / self.sdX

        return {'alpha': alpha0, 'beta': beta0, 'h': h, 'res': res, 'nirw': count, 'lambda': Lambda}

    def cqr_l1_path(self, lambda_seq, tau, h=None, kernel="Laplacian", \
                    order="ascend", standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized Conquer

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        order : a character string indicating the order of lambda values along which the solution path is obtained; default is 'ascend'.

        Returns
        -------
        'beta_seq' : a sequence of l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of residual vectors.

        'size_seq' : a sequence of numbers of selected variables.

        'lambda_seq' : a sequence of lambda values in ascending/descending order.

        'bw' : bandwidth.
        '''
        if h == None: h = self.bandwidth(max(tau))

        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        alpha_seq = np.zeros(shape=(len(tau), len(lambda_seq)))
        beta_seq = np.zeros(shape=(self.X.shape[1], len(lambda_seq)))
        res_seq = np.zeros(shape=(self.n, len(lambda_seq)))
        model = self.cqr_l1(Lambda=lambda_seq[0], tau=tau, h=h, kernel=kernel, standardize=standardize, adjust=False)
        alpha_seq[:, 0], beta_seq[:, 0], res_seq[:, 0] = model['alpha'], model['beta'], model['res']

        for l in range(1, len(lambda_seq)):
            model = self.cqr_l1(lambda_seq[l], tau, h, kernel, alpha_seq[:, l - 1], beta_seq[:, l - 1],
                                res_seq[:, l - 1], standardize,
                                adjust=False)
            beta_seq[:, l], res_seq[:, l] = model['beta'], model['res']

        if standardize and adjust:
            beta_seq[:, ] = beta_seq[:, ] / self.sdX[:, None]

        return {'alpha_seq': alpha_seq, 'beta_seq': beta_seq, 'res_seq': res_seq,
                'size_seq': np.sum(beta_seq[:, :] != 0, axis=0),
                'lambda_seq': lambda_seq, 'bw': h}

    def cqrp_admm_path(self, lambda_seq, tau, order="ascend", sg=0.03, maxit=20000):
        '''
            Solution Path of L1-Penalized CQR via ADMM

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        order : a character string indicating the order of lambda values along which the solution path is obtained; default is 'ascend'.

        Returns
        -------
        'beta_seq' : a sequence of l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'size_seq' : a sequence of numbers of selected variables.

        'lambda_seq' : a sequence of lambda values in ascending/descending order.

        '''


        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        alpha_seq = np.zeros(shape=(len(tau), len(lambda_seq)))
        beta_seq = np.zeros(shape=(self.X.shape[1], len(lambda_seq)))

        model = self.cqrp_admm_smw(Lambda=lambda_seq[0], tau=tau)
        alpha_seq[:, 0], beta_seq[:, 0] = model['alpha'], model['beta']

        for l in range(1, len(lambda_seq)):
            model = self.cqrp_admm_smw(lambda_seq[l], tau=tau, alpha0=alpha_seq[:, l - 1], beta0=beta_seq[:, l - 1], sg=sg, maxit=maxit)
            alpha_seq[:,l], beta_seq[:, l] = model['alpha'], model['beta']

        return {'alpha_seq': alpha_seq, 'beta_seq': beta_seq,
                'size_seq': np.sum(beta_seq[:, :] != 0, axis=0),
                'lambda_seq': lambda_seq}

    def cqrp_admm_path_nodemean(self, lambda_seq, tau, order="ascend", sg=0.03, maxit=20000):
        '''
            Solution Path of L1-Penalized Conquer

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        order : a character string indicating the order of lambda values along which the solution path is obtained; default is 'ascend'.

        Returns
        -------
        'beta_seq' : a sequence of l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'size_seq' : a sequence of numbers of selected variables.

        'lambda_seq' : a sequence of lambda values in ascending/descending order.

        '''


        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        alpha_seq = np.zeros(shape=(len(tau), len(lambda_seq)))
        beta_seq = np.zeros(shape=(self.X.shape[1], len(lambda_seq)))

        model = self.cqrp_admm(Lambda=lambda_seq[0], tau=tau)
        alpha_seq[:, 0], beta_seq[:, 0] = model['alpha'], model['beta']

        for l in range(1, len(lambda_seq)):
            model = self.cqrp_admm(lambda_seq[l], tau=tau, alpha0=alpha_seq[:, l - 1], beta0=beta_seq[:, l - 1])
            alpha_seq[:,l], beta_seq[:, l] = model['alpha'], model['beta']

        return {'alpha_seq': alpha_seq, 'beta_seq': beta_seq,
                'size_seq': np.sum(beta_seq[:, :] != 0, axis=0),
                'lambda_seq': lambda_seq}


    def cqr_irw_path(self, lambda_seq, tau=np.array([]), h=None, kernel="Laplacian", order="ascend",
                     penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Solution Path of Iteratively Reweighted L1-Conquer

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        tau : quantile level; default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.bandwidth().

        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        order : a character string indicating the order of lambda values along which the solution path is obtained; default is 'ascend'.

        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".

        a : the constant (>2) in the concave penality; default is 3.7.

        nstep : number of iterations/steps of the IRW algorithm; default is 5.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        adjust : logical flag for returning coefficients on the original scale; default is TRUE.


        Returns
        -------
        'beta_seq' : a sequence of irw-l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of residual vectors.

        'size_seq' : a sequence of numbers of selected variables.

        'lambda_seq' : a sequence of lambda values in ascending/descending order.

        'bw' : bandwidth.
        '''
        if h == None: h = self.bandwidth(max(tau))

        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        alpha_seq = np.zeros(shape=(len(tau), len(lambda_seq)))
        beta_seq = np.zeros(shape=(self.X.shape[1], len(lambda_seq)))
        res_seq = np.zeros(shape=(self.n, len(lambda_seq)))

        model = self.cqr_irw(lambda_seq[0], tau, h, kernel, penalty=penalty, a=a, nstep=nstep,
                             standardize=standardize, adjust=False)
        alpha_seq[:, 0], beta_seq[:, 0], res_seq[:, 0] = model['alpha'], model['beta'], model['res']

        for l in range(1, len(lambda_seq)):
            model = self.cqr_irw(lambda_seq[l], tau, h, kernel, alpha_seq[:, l - 1], beta_seq[:, l - 1],
                                 res_seq[:, l - 1],
                                 penalty, a, nstep, standardize, adjust=False)
            alpha_seq[:, l], beta_seq[:, l], res_seq[:, l] = model['alpha'], model['beta'], model['res']

        if standardize and adjust:
            beta_seq[:, ] = beta_seq[:, ] / self.sdX[:, None]

        return {'alpha_seq': alpha_seq, 'beta_seq': beta_seq, 'res_seq': res_seq,
                'size_seq': np.sum(beta_seq[:, :] != 0, axis=0),
                'lambda_seq': lambda_seq, 'bw': h}

    def cqr_bic(self, tau, h=None, lambda_seq=np.array([]), nlambda=35, \
                kernel="Laplacian", order='ascend', max_size=False, Cn=None, \
                penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Model Selection via Bayesian Information Criterion

        Reference
        ---------
        Model selection via Bayesian information criterion for quantile regression models (2014)
        by Eun Ryung Lee, Hohsuk Noh and Byeong U. Park
        Journal of the American Statistical Association 109(505): 216--229.

        Arguments
        ---------
        see l1_path() and irw_path()

        max_size : an upper bound on the selected model size; default is FALSE (no size restriction).

        Cn : a positive constant (that diverges as sample size increases) in the modified BIC; default is log(p).

        Returns
        -------
        'bic_beta' : estimated coefficient vector for the BIC-selected model.

        'bic_seq' : residual vector for the BIC-selected model.

        'bic_size' : size of the BIC-selected model.

        'bic_lambda' : lambda value that corresponds to the BIC-selected model.

        'beta_seq' : a sequence of penalized conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'size_seq' : a vector of estimated model sizes corresponding to lambda_seq.

        'lambda_seq' : a vector of lambda values.

        'bic' : a vector of BIC values corresponding to lambda_seq.

        'bw' : bandwidth.
        '''
        K = len(tau)
        if standardize:
            X = self.X1
        else:
            X = self.X

        XX = np.tile(X.T, K)

        if not lambda_seq.any():
            sim_lambda = self.cqr_self_tuning(XX, tau=tau)
            lambda_seq = np.linspace(np.quantile(0.5*sim_lambda, 0.95), 4 * np.quantile(sim_lambda, 0.95), num=nlambda)
        else:
            nlambda = len(lambda_seq)

        if Cn == None: Cn = np.log(np.log(self.n))

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD, MCP or CapppedL1")

        if penalty == "L1":
            model_all = self.cqr_l1_path(lambda_seq, tau, h, kernel, order, standardize, adjust)
        else:
            model_all = self.cqr_irw_path(lambda_seq, tau, h, kernel, order, penalty, a, nstep, standardize, adjust)

        BIC = np.array([np.log(self.cqr_check_sum(model_all['res_seq'][:, l], tau, alpha=model_all['alpha_seq'][:, l])) for l in range(0, nlambda)])
        BIC += model_all['size_seq'] * np.log(self.p) * Cn / (2.25 * self.n)
        if not max_size:
            bic_select = BIC == min(BIC)
        else:
            bic_select = BIC == min(BIC[model_all['size_seq'] <= max_size])

        return {'bic_beta': model_all['beta_seq'][:, bic_select],
                'bic_res': model_all['res_seq'][:, bic_select],
                'bic_size': model_all['size_seq'][bic_select],
                'bic_lambda': model_all['lambda_seq'][bic_select],
                'beta_seq': model_all['beta_seq'],
                'size_seq': model_all['size_seq'],
                'lambda_seq': model_all['lambda_seq'],
                'bic': BIC,
                'bw': model_all['bw'],
                'bic_select_index': bic_select}

    def cqrp_admm_bic(self, tau, lambda_seq=np.array([]), nlambda=100,
                order='ascend', max_size=False, Cn=None, sg=0.03):

        K = len(tau)
        X = self.X
        XX = np.tile(X.T, K)

        if not lambda_seq.any():
            sim_lambda = self.cqr_self_tuning(XX, tau=tau)
            lambda_seq = np.linspace(0.5*np.quantile(sim_lambda, 0.95), 4 * np.quantile(sim_lambda, 0.95), num=nlambda)
        else:
            nlambda = len(lambda_seq)

        if Cn == None: Cn = np.log(np.log(self.n))

        model_all = self.cqrp_admm_path(lambda_seq, tau, order, sg=sg)

        BIC = np.array([np.log(self.cqr_check_sum(self.Y-X.dot(model_all['beta_seq'][:,l]), tau, alpha=model_all['alpha_seq'][:, l])) for l in range(0, nlambda)])
        BIC += model_all['size_seq'] * np.log(self.p) * Cn / (2.25*self.n)
        if not max_size:
            bic_select = BIC == min(BIC)
        else:
            bic_select = BIC == min(BIC[model_all['size_seq'] <= max_size])

        return {'bic_beta': model_all['beta_seq'][:, bic_select],

                'bic_size': model_all['size_seq'][bic_select],
                'bic_lambda': model_all['lambda_seq'][bic_select],
                'beta_seq': model_all['beta_seq'],
                'size_seq': model_all['size_seq'],
                'lambda_seq': model_all['lambda_seq'],
                'bic': BIC,
                'bic_select_index': bic_select}

    def cqr_bic_highdim(self, tau, h=None, lambda_seq=np.array([]), nlambda=100, \
                kernel="Laplacian", order='ascend', max_size=False, Cn=None, \
                penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Model Selection via Bayesian Information Criterion

        Reference
        ---------
        Model selection via Bayesian information criterion for quantile regression models (2014)
        by Eun Ryung Lee, Hohsuk Noh and Byeong U. Park
        Journal of the American Statistical Association 109(505): 216--229.

        Arguments
        ---------
        see l1_path() and irw_path()

        max_size : an upper bound on the selected model size; default is FALSE (no size restriction).

        Cn : a positive constant (that diverges as sample size increases) in the modified BIC; default is log(p).

        Returns
        -------
        'bic_beta' : estimated coefficient vector for the BIC-selected model.

        'bic_seq' : residual vector for the BIC-selected model.

        'bic_size' : size of the BIC-selected model.

        'bic_lambda' : lambda value that corresponds to the BIC-selected model.

        'beta_seq' : a sequence of penalized conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'size_seq' : a vector of estimated model sizes corresponding to lambda_seq.

        'lambda_seq' : a vector of lambda values.

        'bic' : a vector of BIC values corresponding to lambda_seq.

        'bw' : bandwidth.
        '''
        K = len(tau)
        if standardize:
            X = self.X1
        else:
            X = self.X

        XX = np.tile(X.T, K)

        if not lambda_seq.any():
            sim_lambda = self.cqr_self_tuning(XX, tau=tau)
            lambda_seq = np.linspace(np.quantile(sim_lambda, 0.5), 8 * np.quantile(sim_lambda, 0.5), num=nlambda)
        else:
            nlambda = len(lambda_seq)

        if Cn == None: Cn = np.log(self.p)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD, MCP or CapppedL1")

        if penalty == "L1":
            model_all = self.cqr_l1_path(lambda_seq, tau, h, kernel, order, standardize, adjust)
        else:
            model_all = self.cqr_irw_path(lambda_seq, tau, h, kernel, order, penalty, a, nstep, standardize, adjust)

        BIC = np.array([self.cqr_check_sum(model_all['res_seq'][:, l], tau, alpha=model_all['alpha_seq'][:, l]) for l in range(0, nlambda)])
        BIC += model_all['size_seq'] * np.log(self.n) * Cn / (2 * self.n)
        if not max_size:
            bic_select = BIC == min(BIC)
        else:
            bic_select = BIC == min(BIC[model_all['size_seq'] <= max_size])

        return {'bic_beta': model_all['beta_seq'][:, bic_select],
                'bic_res': model_all['res_seq'][:, bic_select],
                'bic_size': model_all['size_seq'][bic_select],
                'bic_lambda': model_all['lambda_seq'][bic_select],
                'beta_seq': model_all['beta_seq'],
                'size_seq': model_all['size_seq'],
                'lambda_seq': model_all['lambda_seq'],
                'bic': BIC,
                'bw': model_all['bw'],
                'bic_select_index': bic_select}

    def ranklasso(self, Lambda = np.array([]), numsim=500, c=1.01, alpha0=0.1,method='revised simplex' ):
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



class cv_lambda():
    '''
        Cross-Validated Penalized Conquer
    '''
    penalties = ["L1", "SCAD", "MCP"]
    opt = {'nsim': 200, 'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-4, 'irw_tol': 1e-4}

    def __init__(self, X, Y, intercept=True, options={}):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)
        self.itcp = intercept
        self.opt.update(options)

    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into V=nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds

    def fit(self, tau=0.5, h=None, lambda_seq=np.array([]), nlambda=40, nfolds=5,
            kernel="Laplacian", penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):

        sqr = high_dim(self.X, self.Y, self.itcp, self.opt)
        if h == None: h = sqr.bandwidth(tau)

        if not lambda_seq.any():
            lambda_max = np.max(sqr.self_tuning(tau, standardize))
            lambda_seq = np.linspace(0.25 * lambda_max, 1.25 * lambda_max, nlambda)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD or MCP")

        check_loss = lambda x: np.mean(np.where(x >= 0, tau * x, (tau - 1) * x))  # empirical check loss
        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx, folds[v]), :], self.Y[np.setdiff1d(idx, folds[v])]
            X_val, Y_val = self.X[folds[v], :], self.Y[folds[v]]
            sqr_train = high_dim(X_train, Y_train, self.itcp, self.opt)

            if penalty == "L1":
                model = sqr_train.l1_path(lambda_seq, tau, h, kernel, 'ascend', standardize, adjust)
            else:
                model = sqr_train.irw_path(lambda_seq, tau, h, kernel, 'ascend', penalty, a, nstep, standardize, adjust)

            val_err[v, :] = np.array([check_loss(Y_val - model['beta_seq'][0, l] * self.itcp \
                                                 - X_val.dot(model['beta_seq'][self.itcp:, l])) for l in
                                      range(nlambda)])

        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        lambda_min = model['lambda_seq'][cv_err == cv_min][0]
        if penalty == "L1":
            cv_model = sqr.l1(lambda_min, tau, h, kernel, standardize=standardize, adjust=adjust)
        else:
            cv_model = sqr.irw(lambda_min, tau, h, kernel, penalty=penalty, a=a, nstep=nstep, \
                               standardize=standardize, adjust=adjust)

        return {'cv_beta': cv_model['beta'],
                'cv_res': cv_model['res'],
                'lambda_min': lambda_min,
                'lambda_seq': model['lambda_seq'],
                'min_cv_err': cv_min,
                'cv_err': cv_err}

class cv_lambda_cqr():
    '''
        Cross-Validated Penalized Composite Conquer
    '''
    penalties = ["L1", "SCAD", "MCP"]
    opt = {'nsim': 200, 'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-4, 'irw_tol': 1e-4}

    def __init__(self, X, Y):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)
        self.sdX=np.std(X, axis=0)


    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into V=nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds

    def cqr_check_sum_cv(self, x, tau, alpha):
        ccs = 0
        for i in range(0, len(tau)):
            ccs = ccs + np.sum(np.where(x - alpha[i] >= 0, tau[i] * (x - alpha[i]), (tau[i] - 1) * (x - alpha[i])))

        return ccs / len(tau)

    def fit(self, tau, h=None, lambda_seq=np.array([]), nlambda=50, nfolds=5,
            kernel="Laplacian", penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        K=len(tau)
        if standardize:
            X = self.X/self.sdX
        else:
            X = self.X
        scqr = high_dim(X, self.Y,intercept=False)


        XX = np.tile(X.T, K)
        if h == None: h = scqr.bandwidth(max(tau))

        if not lambda_seq.any():
            lambda_med = np.quantile(scqr.cqr_self_tuning(XX,tau),0.5)
            lambda_seq = np.linspace(lambda_med, 8 * lambda_med, nlambda)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD or MCP")

        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx, folds[v]), :], self.Y[np.setdiff1d(idx, folds[v])]
            X_val, Y_val = self.X[folds[v], :], self.Y[folds[v]]
            scqr_train = high_dim(X_train, Y_train, intercept=False)

            if penalty == "L1":
                model = scqr_train.cqr_l1_path(lambda_seq, tau, h, kernel, 'ascend', standardize, adjust)
            else:
                model = scqr_train.cqr_irw_path(lambda_seq, tau, h, kernel, 'ascend', penalty, a, nstep, standardize, adjust)

            val_err[v, :] = np.array([self.cqr_check_sum_cv(Y_val -X_val.dot(model['beta_seq'][:, l]), tau, alpha=model['alpha_seq'][:, l]) for l in range(nlambda)])

        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        lambda_min = model['lambda_seq'][cv_err == cv_min][0]
        if penalty == "L1":
            cv_model = scqr.cqr_l1(Lambda=lambda_min, tau=tau, h=h, kernel=kernel, standardize=standardize, adjust=adjust)
        else:
            cv_model = scqr.cqr_irw(Lambda=lambda_min, tau=tau, h=h, kernel=kernel, penalty=penalty, a=a, nstep=nstep,
                               standardize=standardize, adjust=adjust)

        return {'cv_beta': cv_model['beta'],
                'cv_res': cv_model['res'],
                'lambda_min': lambda_min,
                'lambda_seq': model['lambda_seq'],
                'min_cv_err': cv_min,
                'cv_err': cv_err}
class cv_lambda_cqrp_admm():
    '''
        Cross-Validated Penalized CQR_ADMM
    '''

    def __init__(self, X, Y):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)

    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into V=nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds

    def cqr_check_sum_cv(self, x, tau, alpha):
        ccs = 0
        for i in range(0, len(tau)):
            ccs = ccs + np.sum(np.where(x - alpha[i] >= 0, tau[i] * (x - alpha[i]), (tau[i] - 1) * (x - alpha[i])))

        return ccs / len(tau)

    def fit(self, tau, lambda_seq=np.array([]), nlambda=50, nfolds=5):
        K=len(tau)
        scqr = high_dim(self.X, self.Y, intercept=False)


        XX = np.tile(self.X.T, K)
        if not lambda_seq.any():
            lambda_med = np.quantile(scqr.cqr_self_tuning(XX,tau),0.5)
            lambda_seq = np.linspace(lambda_med, 8 * lambda_med, nlambda)
        else:
            nlambda = len(lambda_seq)

        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx, folds[v]), :], self.Y[np.setdiff1d(idx, folds[v])]
            X_val, Y_val = self.X[folds[v], :], self.Y[folds[v]]
            scqr_train = high_dim(X_train, Y_train, intercept=False)
            model = scqr_train.cqrp_admm_path_nodemean(lambda_seq, tau=tau, order='ascend')


            val_err[v, :] = np.array([self.cqr_check_sum_cv(Y_val -X_val.dot(model['beta_seq'][:, l]), tau, alpha=model['alpha_seq'][:, l]) for l in range(nlambda)])

        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        lambda_min = model['lambda_seq'][cv_err == cv_min][0]
        cv_model = scqr.cqrp_admm_smw(Lambda=lambda_min, tau=tau)
        return {'cv_beta': cv_model['beta'],

                'lambda_min': lambda_min,
                'lambda_seq': model['lambda_seq'],
                'min_cv_err': cv_min,
                'cv_err': cv_err}


class cv_lambda_cqrp_admm_fast():
    '''
        Cross-Validated Penalized CQR_ADMM
    '''

    def __init__(self, X, Y, truebeta=None):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)
        self.truebeta = truebeta

    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into V=nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds

    def cqr_check_sum_cv(self, x, tau, alpha):
        ccs = 0
        for i in range(0, len(tau)):
            ccs = ccs + np.sum(np.where(x - alpha[i] >= 0, tau[i] * (x - alpha[i]), (tau[i] - 1) * (x - alpha[i])))

        return ccs / len(tau)

    def fit(self, tau, lambda_seq=np.array([]), nlambda=50 , nfolds=5 ):
        K = len(tau)
        scqr = high_dim(self.X, self.Y, intercept=False)

        XX = np.tile(self.X.T, K)
        if not lambda_seq.any():
            lambda_med = np.quantile(scqr.cqr_self_tuning(XX, tau), 0.5)
            lambda_seq = np.linspace(lambda_med, 8 * lambda_med, nlambda)
        else:
            nlambda = len(lambda_seq)

        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx, folds[v]), :]-self.X[np.setdiff1d(idx, folds[v]), :].mean(axis=0), self.Y[np.setdiff1d(idx, folds[v])]-self.X[np.setdiff1d(idx, folds[v]), :].mean(axis=0).dot(self.truebeta)
            X_val, Y_val = self.X[folds[v], :]-self.X[folds[v], :].mean(axis=0), self.Y[folds[v]]-self.X[folds[v], :].mean(axis=0).dot(self.truebeta)
            scqr_train = high_dim(X_train, Y_train, intercept=False)
            model = scqr_train.cqrp_admm_path(lambda_seq, tau=tau, order='ascend')

            val_err[v, :] = np.array(
                [self.cqr_check_sum_cv(Y_val - X_val.dot(model['beta_seq'][:, l]), tau, alpha=model['alpha_seq'][:, l])
                 for l in range(nlambda)])

        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        lambda_min = model['lambda_seq'][cv_err == cv_min][0]
        cv_model = scqr.cqrp_admm_smw(Lambda=lambda_min, tau=tau)
        return {'cv_beta': cv_model['beta'],

                'lambda_min': lambda_min,
                'lambda_seq': model['lambda_seq'],
                'min_cv_err': cv_min,
                'cv_err': cv_err}
