import numpy as np

from ..regression_base import Regression


class Non_Linear_Regression(Regression):
    """
    Non-Linear Regression model.
    """

    def __init__(self, degree=2):
        self.theta = None  # Parameters vector
        self.degree = degree  # Degree of the polynomial

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the non-linear regression model to the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Create polynomial features
        X_design = self.create_design_matrix(X)
        self.theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y

    def create_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Create a design matrix with polynomial features.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        X_poly : array-like, shape (n_samples, n_polynomial_features)
            Feature matrix with polynomial features.
        """
        x_lenght, _ = X.shape
        X_poly = np.zeros((x_lenght, self.degree))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly

    def predict(self, X: np.ndarray):
        X_design = self.create_design_matrix(X)
        return X_design @ self.theta  # Return the predicted values
