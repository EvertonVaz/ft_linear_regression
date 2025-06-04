import pytest
from src.ft_linear_regression import FtLinearRegression as lr

@pytest.fixture
def model():
    """
    Fixture to create an instance of the FTLinearRegression class.
    """
    return lr(learning_rate=0.01, n_iterations=500)

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    return [[1, 2], [3, 4], [5, 6]]


def test_predict_in_same_train_data(model, sample_data):
    """
    Test the predict method of the FTLinearRegression class.
    """
    model.data = sample_data
    X = [row[0] for row in sample_data]
    y = [row[1] for row in sample_data]

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert sum(abs(predictions - y)) < 0.1
    assert all(isinstance(pred, float) for pred in predictions)
    assert all(pred >= 0 for pred in predictions)


def test_predict_in_another_data(model, sample_data):
    """
    Test the predict method of the FTLinearRegression class.
    """
    model.data = sample_data
    X = [row[0] for row in sample_data]
    y = [row[1] for row in sample_data]
    data_to_predict = [6, 7, 8]
    value_to_predict = [7, 8, 9]
    model.fit(X, y)

    predictions = model.predict(data_to_predict)
    assert len(predictions) == len(X)
    assert sum(abs(predictions - value_to_predict)) < 0.1
    assert all(isinstance(pred, float) for pred in predictions)
    assert all(pred >= 0 for pred in predictions)