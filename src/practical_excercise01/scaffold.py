import numpy as np

# DO NOT CHANGE THE NAME OF THE CLASS IF YOU WANT TO SUBMIT 
# TO THE BENCHMARK ONLINE.
class Regression:
    def __init__(self, stepsize=1, max_dim=1, lmbda=0.01):
        self.theta = None # The parameters (weights) of the model
        self.stepsize = stepsize
        self.max_dim = max_dim
        self.lmbda = lmbda

    def _normalize(self, X):
        std = X.std
        mean = X.mean
        X = X - (np.ones(X.shape) * std)
        return X / mean

    def fit(self, X, y):
        """
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        #X = self._normalize(X)
        X_design = self._generate_higher_dim_design_matrix(X,self.max_dim, self.stepsize)
        L2_inv = self._apply_L2(X_design, self.lmbda)
        self.theta = L2_inv @ X_design.T @ y
        return self

    def predict(self, X):
        """
        X: numpy array of shape (n_samples, n_features)

        Return:
            numpy array or list of shape (n_samples,)
        """
        #X = self._normalize(X)
        X_pred = self._generate_higher_dim_design_matrix(X, self.max_dim,self.stepsize)
        return X_pred @ self.theta


    def _generate_higher_dim_design_matrix(self,X,max_dim,stepsize):
        dim = X.shape[0]
        bias = np.ones((dim,1))
        for power in range(1,max_dim+1,stepsize):
            bias = np.hstack((bias,X ** power))
        return bias


    def _apply_L2(self, X, lmbda):
        X_design = X.T @ X
        dim = X_design.shape[0]
        I = np.identity(dim)
        I = I * lmbda
        return np.linalg.inv(X_design + I)

