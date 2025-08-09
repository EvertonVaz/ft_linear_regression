# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/05 19:26:19 by egeraldo          #+#    #+#              #
#    Updated: 2025/08/09 12:44:36 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import json
import numpy as np


class Model:
    X: np.ndarray = np.array([0])
    X_mean: float = 0.0
    X_std: float = 1.0

    y: np.ndarray = np.array([0])
    y_mean: float = 0.0
    y_std: float = 1.0

    theta0: float = 0.0
    theta1: float = 0.0

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        file_path: str = "./model.json",
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.file_path = file_path

    def save_model(self):
        """
        Saves the model parameters to a file.
        :param file_path: The path to the file where the model will be saved.
        """
        with open(self.file_path, "w") as file:
            file.write(f"{self}")

    def load_model(self):
        """
        Loads the model parameters from a file.
        :param file_path: The path to the file from which the model will be loaded.
        """
        if not self.file_path.endswith(".json"):
            raise ValueError("File must be a JSON file with .json extension")
        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.file_path} not found.")
        except json.JSONDecodeError:
            raise ValueError(
                f"File {self.file_path} is not a valid JSON file."
            )

    def __str__(self):
        """
        Returns a string representation of the model parameters.
        :return: A string containing the model parameters.
        """
        return json.dumps(
            {
                "X_mean": self.X_mean,
                "X_std": self.X_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "theta0": self.theta0,
                "theta1": self.theta1,
                "learning_rate": self.learning_rate,
                "n_iterations": self.n_iterations,
            },
            indent=4,
        )


if __name__ == "__main__":
    model = Model()
    print(model)
    model.save_model()
    model.load_model()
    print(model)
