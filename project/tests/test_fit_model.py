import pytest
from src.ft_linear_regression import FtLinearRegression as lr

@pytest.fixture
def model():
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


def test_fit_intern_model(model, sample_data):
    """
    Test the fit_model method of the FTLinearRegression class.
    """
    model.data = sample_data

    model.fit_intern_data()

    assert model.theta0 == 0.0
    assert round(abs(model.theta1 - 0.633), 6) < 0.001
    assert len(model.data) > 0

def test_fit_model(model, sample_data):
    """
    Test the fit_model method of the FTLinearRegression class.
    """
    model.data = sample_data
    X = [row[0] for row in sample_data]
    y = [row[1] for row in sample_data]

    model.fit(X, y)

    assert model.theta0 == 0.0
    assert round(abs(model.theta1 - 0.633), 6) < 0.001
    assert len(model.data) > 0

