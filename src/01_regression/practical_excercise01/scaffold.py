import numpy as np

# DO NOT CHANGE THE NAME OF THE CLASS IF YOU WANT TO SUBMIT 
# TO THE BENCHMARK ONLINE.
class Regression:
    def __init__(self, degrees=[1,4], lmbda=0.001, normalize=False) :
        self.theta = None
        self.degrees = degrees
        self.lmbda = lmbda
        self.mean = None
        self.std = None
        self.normalize = normalize

    def _normalize(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Prevent division by zero for constant features
            self.std = np.where(self.std == 0, 1.0, self.std)
        return (X - self.mean) / self.std

    def fit(self, X, y):
        if self.normalize:
            X = self._normalize(X, fit=True)
        y = np.asarray(y, dtype=float)
        X_design = self._generate_higher_dim_design_matrix(X)
        L2_inv = self._apply_L2(X_design, self.lmbda)
        self.theta = L2_inv @ X_design.T @ y
        return self

    def predict(self, X):
        if self.normalize:
            X = self._normalize(X, fit=False)
        X_pred = self._generate_higher_dim_design_matrix(X)
        return X_pred @ self.theta


    def _generate_higher_dim_design_matrix(self,X):
        dim = X.shape[0]
        bias = np.ones((dim,1))
        for power in self.degrees:
            bias = np.hstack((bias,X ** power))
        return bias
    

    def _apply_L2(self, X, lmbda):
        X_design = X.T @ X
        dim = X_design.shape[0]
        I = np.identity(dim)
        I = I * lmbda
        I[0,0] = 0
        return np.linalg.inv(X_design + I)

