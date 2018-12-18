from recommender.content_based_model import cost_function

import numpy as np


def test_get_cost_function():
    X = np.array([[1.048686, -0.400232, 1.194119],
                  [0.780851, -0.385626,0.521198],
                  [0.641509, -0.547854, -0.083796],
                  [0.453618, -0.800218, 0.680481],
                  [0.937538, 0.106090, 0.361953]])
    theta =  np.array([[0.28544, -1.68427, 0.26294],
                  [0.50501, -0.45465, 0.31746],
                  [-0.43192, -0.47880, 0.84671],
                  [0.72860, -0.27189, 0.32684]])

    Y = np.array([[5, 4, 0, 0],
                  [3, 0, 0, 0],
                  [4, 0, 0, 0],
                  [3, 0, 0, 0],
                  [3, 0, 0, 0]])

    J = cost_function.get_cost_function(X, theta, Y, 0)
    assert round(J, 2) == 22.22