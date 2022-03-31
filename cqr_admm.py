import numpy as np
import numpy.random as rgt
import math

class high_dim():
    '''
        Regularized Convolution Smoothed Composite Quantile Regression via ILAMM
                        (iterative local adaptive majorize-minimization)
    '''
    weights = ['Multinomial', 'Exponential', 'Rademacher']
    penalties = ["L1", "SCAD", "MCP", "CapppedL1"]
    opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-5,
           'irw_tol': 1e-5, 'nsim': 200, 'nboot': 200}

    def __init__(self, X, Y, intercept=True,options={}):

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



    def cqr_check_sum(self, x, tau, alpha):
        ccs = 0
        for i in range(0, len(tau)):
            ccs = ccs + np.sum(np.where(x - alpha[i] >= 0, tau[i] * (x - alpha[i]), (tau[i] - 1) * (x - alpha[i])))

        return ccs / len(tau)

    def cqrprox(self, v, a, tau):
        return v-np.maximum((tau-1)/a, np.minimum(v, tau/a))

    def cqr_conquer_lambdasim(self, tau=np.array([])):
        cqr_lambda = (rgt.uniform(0, 1, self.n) <= tau[0]) - tau[0]
        for i in range(1, len(tau)):
            cqr_lambda = np.hstack((cqr_lambda, (rgt.uniform(0, 1, self.n) <= tau[i]) - tau[i]))
        return cqr_lambda / (len(tau) * self.n)

    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x) * np.where(tmp <= 0, 0, tmp)


    def cqr_self_tuning(self, XX, tau=np.array([])):
        cqr_lambda_sim = np.array([max(abs(XX.dot(self.cqr_conquer_lambdasim(tau))))
                                   for b in range(self.opt['nsim'])])
        return 2*cqr_lambda_sim


    def cqrp_admm(self, Lambda=np.array([]), tau=np.array([]), sg=0.01, alpha0=np.array([]),
               beta0=np.array([]), e1=1e-3, e2=1e-3,maxit=20000, lambdaparameter=1.3):
        '''
        ADMM based algorithm for CQR

        Reference
        ---------
        Sparse composite quantile regression in ultrahigh dimensions with tuning parameter calibration.
        by Yuwen Gu and Hui Zou
        IEEE Transactions on Information Theory 66: 7132--7154.

        '''

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
        '''
            ADMM based algorithm for CQR.
            Sherman-Morrison-Woodbury formula has been applied.
            Demeaning the design matrix is required for this algorithm to run faster

            Reference
            ---------
            Sparse composite quantile regression in ultrahigh dimensions with tuning parameter calibration.
            by Yuwen Gu and Hui Zou
            IEEE Transactions on Information Theory 66: 7132--7154.

        '''
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
        '''
            ADMM based algorithm for CQR.
            Algorithm with out demeaning the design matrix.
            Reference
            ---------
            Sparse composite quantile regression in ultrahigh dimensions with tuning parameter calibration.
            by Yuwen Gu and Hui Zou
            IEEE Transactions on Information Theory 66: 7132--7154.

        '''
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
            Solution Path of L1-Penalized CQR via ADMM, without demeaning

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
