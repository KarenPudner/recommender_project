import numpy as np


def get_cost_function(film_features, user_parameters, Y, lambda_value):
    ratings = np.matmul(film_features, np.transpose(user_parameters))
    error = np.subtract(ratings, Y)

    Y[Y > 0] = 1

    print(Y)
    print(error)
    error_factor = np.multiply(error, Y)
    print(error_factor)
    error_squared = np.multiply(error_factor, error_factor)
    print(error_squared)
    J = ((sum(sum(error_squared))) / 2)

    # regularised_X = sum(sum(X. ^ 2)) * (lambda / 2);
    #                                            regularised_Theta=sum(sum(Theta.^ 2)) * ( lambda / 2);
    #                                            J=((sum(sum(error_squared))) / 2)+ regularised_X + regularised_Theta;
    #
    #
    #                                            X_grad = error_factor * Theta + X.* lambda;
    #                                            Theta_grad = error_factor' * X + Theta.*lambda;
    return J