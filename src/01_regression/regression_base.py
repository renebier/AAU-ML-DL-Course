from abc import ABC, abstractmethod

import numpy as np


class Regression(ABC):
    """
    Abstract base class for regression models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the regression model to the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        raise NotImplementedError(
            "The 'fit' method must be implemented in the subclass."
        )

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        Predict using the regression model.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns:
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        raise NotImplementedError(
            "The 'predict' method must be implemented in the subclass."
        )
