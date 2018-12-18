import numpy as np


def get_cost_function(film_features, user_parameters, Y, lambda_value):


    ratings = np.matmul(film_features, np.transpose(user_parameters))
    error = np.subtract(ratings, Y)

    R = np.where(Y > 0, 1, Y)

    error_factor = np.multiply(error, R)
    error_squared = np.square(error_factor)

    regularised_film_features = ((np.square(film_features)).sum()) * (lambda_value/2)
    regularised_user_parameters = ((np.square(user_parameters)).sum()) * (lambda_value / 2)

    J = ((sum(sum(error_squared))) / 2) + regularised_film_features + regularised_user_parameters

    film_features_grad = np.matmul(error_factor, user_parameters) + np.multiply(film_features, lambda_value)
    user_parameters_grad = np.matmul(np.transpose(error_factor), film_features) + np.multiply(user_parameters, lambda_value)

    return [J, film_features_grad, user_parameters_grad]