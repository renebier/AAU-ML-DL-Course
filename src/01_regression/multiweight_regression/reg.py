import numpy as np

from ..regression_base import Regression


class Multiweight_Linear_Regression(Regression):
    """
    Multiweight Linear Regression model.
    """

    def __init__(self):
        self.theta = None  # Parameters vector

    def fit(self, X, y):
        """
        Fit the multiweight regression model to the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        bias = np.ones((X.shape[0], 1))  # Create a bias term (intercept)
        X_design = np.hstack((bias, X))  # Add the bias term to the input features

        XtX_inv = np.linalg.inv(X_design.T @ X_design)  # Inverse of X^T * X
        self.theta = XtX_inv @ X_design.T @ y

    def predict(self, X):
        """
        Predict using the multiweight regression model.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns:
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        # Add a bias term (intercept) to the input features
        bias = np.ones((X.shape[0], 1))  # Create a bias term (intercept)
        X_design = np.hstack((bias, X))  # Add the bias term to the input features
        return X_design @ self.theta  # Return the predicted values
