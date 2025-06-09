# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    bonus.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/06 22:43:37 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/09 21:17:20 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.ft_linear_regression import FtLinearRegression as lr


def bonus() -> None:
    """
    Bonus task for the 42 Linear Regression project.
    """
    model_path = input(
        "Enter the path to save the model (default: ./model.json): "
    )
    if not model_path:
        model_path = "./model.json"

    file_path = input(
        "Enter the path to the CSV file (default: src/data.csv): "
    )
    if not file_path:
        file_path = "src/data.csv"

    try:
        linear = lr(model_path=model_path)
        linear.read_csv(file_path)
        linear.fit_intern_data()
        linear.plot()
    except Exception as e:
        return print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        bonus()
        print("Bonus task completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
