# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    estimatePrice.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/05 20:45:08 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 23:32:06 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.ft_linear_regression import FtLinearRegression as lr


def estimatePrice() -> list[float]:
    """
    Predicts the output for a given input feature using the trained linear regression model.

    :return: A list containing the predicted output.
    """
    model_path = input("Enter the path to the model (default: ./model.json): ")
    if not model_path:
        model_path = "./model.json"

    linear = lr(model_path=model_path)

    X = input("Enter the mileage: ")
    if not X:
        X = 0.0
    X = float(X)
    return linear.predict([X])


if __name__ == "__main__":
    try:
        pred = estimatePrice()[0]
        print(f"Price estimate: {pred}")
    except Exception as e:
        print(f"An error occurred: {e}")
