import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from . import Multiweight_Linear_Regression

if __name__ == "__main__":
    # Load the California housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Print the shape of the dataset
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    regression_model = Multiweight_Linear_Regression()
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)
    mean_squared_error = ((y_test - y_pred) ** 2).mean()
    print(f"Mean Squared Error: {mean_squared_error}")
    absolute_errors = abs(y_test - y_pred)
    print(f"Mean Absolute Error: {absolute_errors.mean()}")
    variance_score = 1 - (np.var(y_test - y_pred) / np.var(y_test))
    print(f"Variance Score: {variance_score}")
    f1_score = (
        2
        * (variance_score * (1 - mean_squared_error))
        / (variance_score + (1 - mean_squared_error))
    )
    print(f"F1 Score: {f1_score}")
