from recommender.content_based_model import cost_function
import pytest

import numpy as np

X = np.array([[1.048686, -0.400232, 1.194119],
              [0.780851, -0.385626, 0.521198],
              [0.641509, -0.547854, -0.083796],
              [0.453618, -0.800218, 0.680481],
              [0.937538, 0.106090, 0.361953]])

theta = np.array([[0.28544, -1.68427, 0.26294],
                  [0.50501, -0.45465, 0.31746],
                  [-0.43192, -0.47880, 0.84671],
                  [0.72860, -0.27189, 0.32684]])

Y = np.array([[5, 4, 0, 0],
              [3, 0, 0, 0],
              [4, 0, 0, 0],
              [3, 0, 0, 0],
              [3, 0, 0, 0]])

X_grad = np.array([[-0.96, 6.97, -0.11],
              [0.60, 2.77, 0.26],
              [0.13, 4.09, -0.89],
              [0.30, 1.06, 0.67],
              [0.60, 4.90, -0.20]])

Theta_grad = np.array([[-10.14, 2.10, -6.77],
              [-2.29, 0.48, -3.00],
              [-0.65, -0.71821, 1.27],
              [1.09, -0.41, 0.49]])


def test_get_cost_function():
    [J, film_features_grad, user_parameters_grad] = cost_function.get_cost_function(X, theta, Y, 0)
    assert round(J, 2) == 22.22


def test_get_cost_function_with_lambda():
    [J, film_features_grad, user_parameters_grad] = cost_function.get_cost_function(X, theta, Y, 1.5)
    assert round(J, 2) == 31.34
    assert np.around(film_features_grad, decimals=2).all() == X_grad.all()
    assert np.around(user_parameters_grad, decimals=2).all() == Theta_grad.all()