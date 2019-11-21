#!/usr/bin/env python3
"""Testing SitkData class to ensure ability to manipulate data representations

    Example:
        This file should be run via the CMake interface, via

            $ make test

"""
import numpy as np
import pytest

import SimpleITK as sitk
from muscledef.preprocess.sitkdata import SitkData


@pytest.fixture
def test_image_with_matrix_data():
    """
    Creates sample object and returns image and matrix_data
    representations
    """
    matrix_data = np.random.randint(255, size=(3, 4, 5))
    image = sitk.GetImageFromArray(matrix_data.T)
    return (image, matrix_data)


@pytest.fixture
def test_file_with_matrix_data():
    """
    Creates a sample object, saves a copy in testdata, and returns
    filename and matrix_data representation

    TODO: Handle testdata folder
        -Hard code data?
        -Make as temp file for data generation?
            -If so, put in gitignore
    """
    filename = "test/testdata/sitkdata_test_image.nii"
    matrix_filename = "test/testdata/sitkdata_test_matrix.npy"

    with open(matrix_filename) as f:
        matrix_data = np.load(matrix_filename)

    return (filename, matrix_data)


@pytest.fixture
def rand_int_matrix():
    """
    Generates a matrix of integers [0, 3)
    """
    return np.random.randint(3, size=(3, 4, 5))


def test_from_file(test_file_with_matrix_data):
    filename, matrix_data = test_file_with_matrix_data
    data = SitkData.from_file(filename)
    assert "image" in data.current
    assert data.d == len(matrix_data.shape)

    assert matrix_data.shape == data.shape
    assert np.isclose(matrix_data, data.matrix_data).all()


def test_from_image(test_image_with_matrix_data):
    image, matrix_data = test_image_with_matrix_data

    data = SitkData.from_image(image)
    assert "image" in data.current
    assert data.d == len(matrix_data.shape)

    assert data.shape == matrix_data.shape
    assert np.isclose(data.matrix_data, matrix_data).all()


def test_from_matrix_data(test_image_with_matrix_data):
    matrix_data = test_image_with_matrix_data[1]
    data = SitkData.from_matrix_data(matrix_data)

    assert "matrix_data" in data.current
    assert data.d == len(matrix_data.shape)

    assert data.shape == matrix_data.shape
    assert (data.matrix_data == matrix_data).all()


def test_copy(test_file_with_matrix_data, test_image_with_matrix_data):
    filename = test_file_with_matrix_data[0]
    data_from_file = SitkData.from_file(filename)
    cp_from_file = data_from_file.copy()

    assert data_from_file.shape == cp_from_file.shape
    assert (data_from_file.matrix_data == cp_from_file.matrix_data).all()

    image = test_image_with_matrix_data[0]
    data_from_image = SitkData.from_image(image)
    cp_from_image = data_from_image.copy()

    assert data_from_image.shape == cp_from_image.shape
    assert (data_from_image.matrix_data == cp_from_image.matrix_data).all()

    matrix_data = test_file_with_matrix_data[1]
    data_from_matrix = SitkData.from_matrix_data(matrix_data)
    cp_from_matrix = data_from_matrix.copy()

    assert data_from_matrix.shape == cp_from_matrix.shape
    assert (data_from_matrix.matrix_data == cp_from_matrix.matrix_data).all()


def test_write_image(test_image_with_matrix_data):
    filename = "test/testdata/sitkdata_write_test_image.nii"
    #TODO: Handle cleanup of written image? Just use hardcoded one?
    image, matrix_data = test_image_with_matrix_data
    data_from_image = SitkData.from_image(image)
    data_from_image.write_image(filename)

    data_from_disk = SitkData.from_file(filename)

    assert data_from_image.shape == data_from_disk.shape
    assert (data_from_image.matrix_data == data_from_disk.matrix_data).all()


def test_mask(rand_int_matrix):
    data = SitkData.from_matrix_data(rand_int_matrix)
    matrix_zeros = rand_int_matrix.copy()
    matrix_zeros[matrix_zeros != 0] = 0

    masked1 = data.copy()
    masked1.mask(lambda x: x == 0)
    assert (masked1.matrix_data == matrix_zeros).all()

    matrix_twos = rand_int_matrix.copy()
    matrix_twos[matrix_twos == 2] = 0
    masked2 = data.copy()
    masked2.mask(lambda x: x != 2)
    assert (matrix_twos == masked2.matrix_data).all()

    matrix_lt_twos = rand_int_matrix.copy()
    matrix_lt_twos[matrix_lt_twos >= 2] = 0
    masked3 = data.copy()
    masked3.mask(lambda x: x < 2)
    assert (masked3.matrix_data == matrix_lt_twos).all()


def test_binary_mask(rand_int_matrix):
    data = SitkData.from_matrix_data(rand_int_matrix)
    matrix_zeros = rand_int_matrix.copy()
    matrix_zeros[matrix_zeros != 0] = 0

    masked1 = data.copy()
    masked1.binary_mask(lambda x: x != 0)
    ref = data.matrix_data.copy()
    ref[data.matrix_data != 0] = 1
    ref[data.matrix_data == 0] = 0
    assert (masked1.matrix_data == ref).all()

    matrix_twos = rand_int_matrix.copy()
    matrix_twos[matrix_twos != 2] = 1
    matrix_twos[matrix_twos == 2] = 0
    masked2 = data.copy()
    masked2.binary_mask(lambda x: x != 2)
    assert (matrix_twos == masked2.matrix_data).all()

    matrix_lt_twos = rand_int_matrix.copy()
    matrix_lt_twos[matrix_lt_twos < 2] = 1
    matrix_lt_twos[matrix_lt_twos >= 2] = 0
    masked3 = data.copy()
    masked3.binary_mask(lambda x: x < 2)
    assert (masked3.matrix_data == matrix_lt_twos).all()


def test_getitem():
    #TODO
    pass


def test_setitem():
    #TODO
    pass


def test_matrix_data_to_image():
    #TODO
    pass


def test_image_to_matrix_data():
    #TODO
    pass


def test_write_image():
    #TODO
    pass
