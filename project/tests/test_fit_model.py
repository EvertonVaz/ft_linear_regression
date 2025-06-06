import pytest
from src.ft_linear_regression import FtLinearRegression as lr

@pytest.fixture
def ft_lr():
    """
    Fixture to create an instance of the FTLinearRegression class.
    """
    return lr(learning_rate=0.01, n_iterations=100)

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    return [[1, 2], [3, 4], [5, 6]]


def test_fit_intern_model(ft_lr, sample_data):
    """
    Test the fit_model method of the FTLinearRegression class.
    """
    ft_lr.data = sample_data

    ft_lr.fit_intern_data()

    assert ft_lr.model.theta0 == 0.0
    assert round(abs(ft_lr.model.theta1 - 0.633), 6) < 0.001
    assert len(ft_lr.data) > 0

def test_fit_model(ft_lr, sample_data):
    """
    Test the fit_model method of the FTLinearRegression class.
    """
    ft_lr.data = sample_data
    X = [row[0] for row in sample_data]
    y = [row[1] for row in sample_data]

    ft_lr.fit(X, y)

    assert ft_lr.model.theta0 == 0.0
    assert round(abs(ft_lr.model.theta1 - 0.633), 6) < 0.001
    assert len(ft_lr.data) > 0

