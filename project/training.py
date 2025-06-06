# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/05 21:05:02 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 22:30:19 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.ft_linear_regression import FtLinearRegression as lr


def training() -> None:
    """
    Trains the linear regression model using data from a CSV file.

    :return: None
    """
    model_path = input(
        "Enter the path to save the model (default: ./model.json): "
    )
    if not model_path:
        model_path = "./model.json"
    model = lr(model_path=model_path)

    file_path = input(
        "Enter the path to the CSV file (default: src/data.csv): "
    )
    if not file_path:
        file_path = "src/data.csv"
    model.read_csv(file_path)
    model.fit_intern_data()
    print("Model trained successfully.")


if __name__ == "__main__":
    training()
