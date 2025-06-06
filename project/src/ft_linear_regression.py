# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ft_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/05/14 21:27:00 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 23:31:15 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import numpy as np
import matplotlib.pyplot as plt

from src.model import Model


class FtLinearRegression:
    """
    A simple linear regression model using gradient descent.
    This class implements a linear regression model that can be trained using
    gradient descent. It supports reading data from a CSV file, normalizing the
    data, fitting the model to the data, making predictions, and evaluating the
    model using metrics like Mean Squared Error (MSE) and R-squared score.
    """

    data = []
    model: Model

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        model_path: str = "./model.json",
    ):
        """
        Initializes the linear regression model with a learning rate and number of iterations.
        :param learning_rate: The step size for gradient descent.
        :param n_iterations: The number of iterations for gradient descent.
        """

        self.model = Model(learning_rate, n_iterations, model_path)
        self.model_path = model_path

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
                    raise ValueError(
                        f"Row {i} has {len(row)} columns, but exactly 2 columns are required for X and y."
                    )
        if X is not None and y is not None:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            self.data = list(zip(X, y))

        self.model.X = np.array([row[0] for row in self.data])
        self.model.y = np.array([row[1] for row in self.data])
        self.model.X_mean = self.model.X.mean()
        self.model.X_std = self.model.X.std()
        self.model.y_mean = self.model.y.mean()
        self.model.y_std = self.model.y.std()
        return (
            (self.model.X - self.model.X_mean) / self.model.X_std,
            (self.model.y - self.model.y_mean) / self.model.y_std,
        )

    def __denormalize(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Converts normalized predictions back to original scale.
        """
        return y_normalized * self.model.y_std + self.model.y_mean

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

        for i in range(self.model.n_iterations):
            predictions = self.model.theta0 + self.model.theta1 * X_normalized
            error = predictions - y_normalized
            if i == 0:
                self.mse(y_normalized, predictions, "Initial")

            tmp_theta0 = self.model.learning_rate * (error.sum() / m)
            tmp_theta1 = self.model.learning_rate * (
                (error * X_normalized).sum() / m
            )

            self.model.theta0 -= tmp_theta0
            self.model.theta1 -= tmp_theta1

        self.mse(y_normalized, predictions, "Final")
        self.model.save_model()

    def fit_intern_data(self) -> None:
        """
        Fits the model to the data using gradient descent.
        """
        X, y = zip(*self.data) if self.data else ([], [])
        self.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Predicts the target variable for given input data using the fitted model.
        :param X: Input data for which predictions are to be made.
        :return: Predicted values.
        :raises ValueError: If the model has not been fitted yet.
        """
        try:
            self.model.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
        X = np.array(X)
        X_normalized = (X - self.model.X_mean) / self.model.X_std
        y_pred_normalized = (
            self.model.theta0 + self.model.theta1 * X_normalized
        )
        return self.__denormalize(y_pred_normalized)

    def mse(self, y_true, y_pred, phase) -> float:
        """
        Calculates the Mean Squared Error (MSE) between true and predicted values.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        print(f"{phase} Mean Squared Error: {mse:.6f}")
        return mse

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
        Plots the predicted values against the actual data and saves the plot as 'plot.png'.
        :param X: Input features.
        :param y: Target variable.
        """
        if X is not None and y is not None:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            self.data = list(zip(X, y))
        x_data = [row[0] for row in self.data]
        y_data = [row[1] for row in self.data]
        predictions = self.predict(x_data)
        r2 = self.r2_score(y_data, predictions)

        plt.scatter(x_data, y_data, color="blue", label="Dados reais")
        plt.title(f"Previsão vs Dados reais (R² = {r2:.2f})")
        plt.plot(x_data, predictions, color="red", label="Previsão")
        plt.legend()
        plt.savefig("plot.png")
        print("Gráfico salvo como 'plot.png'")
