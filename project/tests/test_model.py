# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test_model.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: egeraldo <egeraldo@student.42sp.org.br>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/06/06 20:24:04 by egeraldo          #+#    #+#              #
#    Updated: 2025/06/06 23:07:34 by egeraldo         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import json
import pytest
from src.model import Model

@pytest.fixture
def file(tmp_path):
    """
    Fixture to create a temporary file for testing.
    :param tmp_path: The temporary path provided by pytest.
    :return: The path to the temporary file.
    """
    return tmp_path / "model.json"

@pytest.fixture
def model(file):
    """
    Fixture to create a Model instance for testing.
    :param file: The temporary file path.
    :return: An instance of the Model class.
    """
    return Model(learning_rate=0.01, n_iterations=1000, file_path=str(file))

def test_model_initialization(model):
    """
    Test the initialization of the Model class.
    :param model: The Model instance created by the fixture.
    """
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.file_path.endswith("model.json")

def test_model_save_and_load(model, file):
    """
    Test saving and loading the model parameters.
    :param model: The Model instance created by the fixture.
    :param file: The temporary file path.
    """
    model.theta0 = 1.0
    model.theta1 = 2.0
    model.save_model()

    new_model = Model(file_path=str(file))
    new_model.load_model()

    assert new_model.theta0 == 1.0
    assert new_model.theta1 == 2.0
    assert new_model.learning_rate == model.learning_rate
    assert new_model.n_iterations == model.n_iterations

def test_model_load_non_json_file(model):
    """
    Test loading a non-JSON file raises a ValueError.
    :param model: The Model instance created by the fixture.
    """
    model.file_path = "non_json_file.txt"
    with pytest.raises(ValueError, match="File must be a JSON file with .json extension"):
        model.load_model()


def test_model_load_file_not_found(model):
    """
    Test loading a non-existent file raises a FileNotFoundError.
    :param model: The Model instance created by the fixture.
    """
    model.file_path = "non_existent_file.json"

    with pytest.raises(FileNotFoundError, match="File .* not found."):
        model.load_model()


def test_model_load_json_decode_error(model, file):
    """
    Test loading a file with invalid JSON raises a JSONDecodeError.
    :param model: The Model instance created by the fixture.
    :param file: The temporary file path.
    """
    with open(file, "w") as f:
        f.write("invalid json")

    with pytest.raises(ValueError, match="File .* is not a valid JSON file."):
        model.load_model()