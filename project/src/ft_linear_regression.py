# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ft_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/05/14 21:27:00 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/03 22:02:11 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import numpy as np
import matplotlib.pyplot as plt


class FtLinearRegression:
    """
    A simple linear regression model using gradient descent.
    This class implements a linear regression model that can be trained using
    gradient descent. It supports reading data from a CSV file, normalizing the
    data, fitting the model to the data, making predictions, and evaluating the
    model using metrics like Mean Squared Error (MSE) and R-squared score.
    Attributes:
        learning_rate (float): The step size for gradient descent.
        n_iterations (int): The number of iterations for gradient descent.
        theta0 (float): The intercept term of the linear regression model.
        theta1 (float): The slope term of the linear regression model.
        data (list): The dataset read from the CSV file.
    """

    X: np.ndarray = np.array([0])
    X_mean: float = 0.0
    X_std: float = 0.0

    y: np.ndarray = np.array([0])
    y_mean: float = 0.0
    y_std: float = 0.0

    theta0 = 0
    theta1 = 0
    data = []

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initializes the linear regression model with a learning rate and number of iterations.
        :param learning_rate: The step size for gradient descent.
        :param n_iterations: The number of iterations for gradient descent.
        """

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def read_csv(self, file_path: str) -> list[list[float]]:
        """
        Reads a CSV file and returns the data as a list of lists.
        :param file_path: The path to the CSV file.
        :return: A list of lists containing the data from the CSV file.
        """
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            next(reader)
            self.data = [list(map(float, row)) for row in reader]

        return self.data

    def __normalize_data(
        self, X=None, y=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalizes the input features and target variable.
        Transform inputs into standard scale.
        :param X: Input features.
        :param y: Target variable.
        :return: Normalized X and y as numpy arrays.

        """
        if X is None or y is None:
            if not self.data:
                raise ValueError("Data must not be empty.")
            for i, row in enumerate(self.data):
                if len(row) != 2:
                    raise ValueError(f"Row {i} has {len(row)} columns, but exactly 2 columns are required for X and y.")
        if X is not None and y is not None:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            self.data = list(zip(X, y))

        self.X = np.array([row[0] for row in self.data])
        self.y = np.array([row[1] for row in self.data])
        self.X_mean = self.X.mean()
        self.X_std = self.X.std()
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        return (
            (self.X - self.X_mean) / self.X_std,
            (self.y - self.y_mean) / self.y_std,
        )

    def __denormalize(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Converts normalized predictions back to original scale.
        """
        return y_normalized * self.y_std + self.y_mean

    def fit(self, X: list[float], y: list[float]) -> None:
        """
        Fits the model to the provided data using gradient descent.
        :param X: Input features.
        :param y: Target variable.
        :raises ValueError: If the lengths of X and y do not match.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.data = list(zip(X, y))
        X_normalized, y_normalized = self.__normalize_data(X, y)
        m = len(X_normalized)

        for i in range(self.n_iterations):
            predictions = self.theta0 + self.theta1 * X_normalized
            error = predictions - y_normalized

            tmp_theta0 = self.learning_rate * (error.sum() / m)
            tmp_theta1 = self.learning_rate * (
                (error * X_normalized).sum() / m
            )

            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1

            if i % 100 == 0 or i == self.n_iterations - 1:
                mse_value = self.mse(y_normalized, predictions)
                print(f"Iteration {i+1}: MSE = {mse_value:.6f}")

    def fit_intern_data(self) -> None:
        """
        Fits the model to the data using gradient descent.
        """
        X, y = self.__normalize_data()
        self.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Predicts the target variable for given input data using the fitted model.
        :param X: Input data for which predictions are to be made.
        :return: Predicted values.
        :raises ValueError: If the model has not been fitted yet.
        """
        self.X = np.array(X)
        X_normalized = (X - self.X_mean) / self.X_std
        y_pred_normalized = self.theta0 + self.theta1 * X_normalized
        return self.__denormalize(y_pred_normalized)

    def mse(self, y_true, y_pred) -> float:
        """
        Calculates the Mean Squared Error (MSE) between true and predicted values.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred) -> float:
        """
        Calculates the R-squared score between true and predicted values.
        """
        if len(y_pred) != len(y_true):
            raise ValueError("X and y must have the same length.")
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def plot(self, X=None, y=None) -> None:
        """
        Plots the data points and the fitted line.
        :param X: Input features.
        :param y: Target variable.
        """
        if X is None or y is None:
            X = self.X
            y = self.y

        plt.scatter(X, y, color="blue", label="Data Points")
        plt.plot(
            X,
            self.theta0 + self.theta1 * ((X - self.X_mean) / self.X_std),
            color="red",
            label="Fitted Line",
        )
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Linear Regression Fit")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pass
