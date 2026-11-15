import numpy as np

from ..regression_base import Regression


class Linear_Regression(Regression):
    """
    Linear Regression model.
    """

    def __init__(self):
        self.w0 = None  # Intercept
        self.w1 = None  # Coefficient for the single feature

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Add a bias term (intercept) to the input features
        self.w1 = self.learn_w1(X, y)
        self.w0 = self.learn_w0(X, y)

    def learn_w1(self, X, y):
        """
        Learn the coefficient (w1) for a single feature using the Normal Equation.

        Parameters:
        X : array-like, shape (n_samples,)
            Training data for a single feature.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Add a bias term (intercept) to the input features
        x_bar = np.mean(X)
        y_bar = np.mean(y)
        numerator = np.sum((X - x_bar) * (y - y_bar))
        denominator = np.sum((X - x_bar) ** 2)

        # Calculate coefficients using the Normal Equation
        return numerator / denominator if denominator != 0 else 0.0

    def learn_w0(self, X, y):
        """
        Learn the intercept (w0) for a single feature using the Normal Equation.

        Parameters:
        X : array-like, shape (n_samples,)
            Training data for a single feature.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Calculate the intercept using the Normal Equation
        return np.mean(y) - self.w1 * np.mean(X)

    def predict(self, X):
        """
        Predict using the linear regression model.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns:
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        # Add a bias term (intercept) to the input features
        return self.w0 + self.w1 * X
