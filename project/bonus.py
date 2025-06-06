# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    bonus.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/06 22:43:37 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 23:31:45 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from src.ft_linear_regression import FtLinearRegression as lr

def bonus() -> None:
    """
    """
    linear = lr()
    linear.read_csv("src/data.csv")
    linear.fit_intern_data()
    linear.plot()


if __name__ == "__main__":
    try:
        bonus()
        print("Bonus task completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")