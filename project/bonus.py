# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    bonus.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/06 22:43:37 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 22:48:23 by egeraldo         ###   ########.fr        #
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
    bonus()
    print("Bonus task completed successfully.")