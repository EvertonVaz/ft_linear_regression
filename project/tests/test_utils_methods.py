# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test_utils_methods.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/06 23:07:42 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 23:07:43 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pytest
from src.ft_linear_regression import FtLinearRegression as lr

@pytest.fixture
def ft_lr():
    """
    Fixture to create an instance of the FTLinearRegression class.
    """
    return lr(learning_rate=0.01, n_iterations=1000)

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    return [[1, 2], [3, 4], [5, 6]]

def test_initialize(ft_lr):
    """
    Test the initialization of the FTLinearRegression class.
    """
    assert ft_lr.data == []
    assert ft_lr.model.learning_rate == 0.01
    assert ft_lr.model.n_iterations == 1000
    assert ft_lr.model.theta0 == 0
    assert ft_lr.model.theta1 == 0
    assert ft_lr.model.X_mean == 0.0
    assert ft_lr.model.y_mean == 0.0
    assert ft_lr.model.X_std == 1.0
    assert ft_lr.model.y_std == 1.0

def test_read_csv(ft_lr):
    """
    Test the read_csv method of the FTLinearRegression class.
    """
    data = ft_lr.read_csv("src/data.csv")
    assert len(data) > 0
    assert isinstance(data, list)

def test_normalize_data(ft_lr):
    """
    Test the __normalize_data method of the FTLinearRegression class.
    """
    X = [1, 3, 5]
    y = [2, 4, 6]
    X, y = ft_lr._FtLinearRegression__normalize_data(X, y)

    assert len(X) == 3
    assert len(y) == 3
    assert ft_lr.model.X_mean == 3.0
    assert ft_lr.model.X_std > 0
    assert ft_lr.model.y_mean == 4.0
    assert ft_lr.model.y_std > 0


def test_normalize_data_empty(ft_lr):
    """
    Test the __normalize_data method with empty data.
    """
    try:
        ft_lr._FtLinearRegression__normalize_data()
    except ValueError as e:
        assert str(e) == "Data must not be empty."

def test_normalize_data_invalid(ft_lr, sample_data):
    """
    Test the __normalize_data method with invalid data.
    """
    ft_lr = lr()
    ft_lr.data = sample_data
    try:
        ft_lr._FtLinearRegression__normalize_data(X=[1, 2], y=[3])  # Invalid lengths
    except ValueError as e:
        assert str(e) == "X and y must have the same length."

    ft_lr.data = [[1, 2], [3, 4], [5]]  # Invalid data structure
    try:
        ft_lr._FtLinearRegression__normalize_data()
    except ValueError as e:
        assert "but exactly 2 columns are required for X and y." in str(e)

def test_denormalize(ft_lr, sample_data):
    """
    Test the __denormalize method of the FTLinearRegression class.
    """
    ft_lr.data = sample_data
    ft_lr.y_mean = 4.0
    ft_lr.y_std = 2.0
    _, y_normalized = ft_lr._FtLinearRegression__normalize_data()
    y_denormalized = ft_lr._FtLinearRegression__denormalize(y_normalized)

    assert len(y_denormalized) == 3
    assert y_denormalized[0] == 2.0
    assert y_denormalized[1] == 4.0
    assert y_denormalized[2] == 6.0