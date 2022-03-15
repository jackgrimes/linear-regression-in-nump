from operator import itemgetter

import numpy as np

from configs import (
    data_path,
    all_column_names,
    float_feature_column_names,
    str_feature_column_names,
    target_name,
    learning_rate,
)
from log import configure_logger

logger = configure_logger(data_path)


def read_data():
    """
    Reads the data in from the avocado.csv file
    np.loadtxt is a bit unforgiving when wanting to read in data with different dtypes, an index column, and
    a header - so I have opted to read in target, float features, and string features separately
    Returns:
        y, float_features, label_features: all numpy arrays
    """

    float_column_indices = [
        {v: k for k, v in all_column_names.items()}[key]
        for key in float_feature_column_names
    ]
    str_column_indices = [
        {v: k for k, v in all_column_names.items()}[key]
        for key in str_feature_column_names
    ]
    target_column_indices = [
        {v: k for k, v in all_column_names.items()}[key] for key in target_name
    ]

    y = np.loadtxt(
        data_path,
        delimiter=",",
        usecols=target_column_indices,
        dtype=float,
        skiprows=1,
    )
    float_features = np.loadtxt(
        data_path,
        delimiter=",",
        usecols=float_column_indices,
        dtype=float,
        skiprows=1,
    )
    label_features = np.loadtxt(
        data_path,
        delimiter=",",
        usecols=str_column_indices,
        dtype=str,
        skiprows=1,
    )
    return y, float_features, label_features


def get_dummies(string_array: int, feature_name: str):
    """
    Custom function for getting dummy variables/one-hot-encoding from a numpy array of strings
    Args:
        string_array: 1-D string array for converting to dummies
        feature_name: name of original feature

    Returns:

    """

    # unique_classes is the unique classes, class_number_each_row is the inverse: the class number for each value
    unique_classes, class_number_each_row = np.unique(string_array, return_inverse=True)

    # dummy_columns is a np array, all zeros, with a column for every class,
    # and a row for every example/row in original data
    dummy_columns = np.zeros(
        (class_number_each_row.size, class_number_each_row.max() + 1)
    )

    # for each row in dummy_columns, set the column corresponding to the value class_number_each_row,
    # equal to 1 (get the dummy variables)
    dummy_columns[np.arange(class_number_each_row.size), class_number_each_row] = 1

    # drop the first dummy variable - otherwise we will have problems with multicollinearity
    dummy_columns = dummy_columns[:, 1:]

    # get feature names for the dummy columns - for use in logging etc later one
    feature_names = [feature_name + "_" + x for x in list(unique_classes)[1:]]

    return dummy_columns, feature_names


def extract_year_and_month(label_features, col_index_of_date):
    """
    Takes the date string column and returns the year and month values, for using as dummy variables
    Args:
        label_features: np array of all the string features
        col_index_of_date: the column index of the date column in the label_features array

    Returns:
        year_and_month_array: array with one column for year, another for month

    """
    # get date column, split on "-"
    dates_as_lists = np.char.split(label_features[:, col_index_of_date], "-")

    # ravel to split list elements over columns, drop day of month here
    year_and_month_array = np.stack(dates_as_lists.ravel())[:, 0:2]

    return year_and_month_array


def get_all_dummies(label_features, year_and_month):
    """
    Get dummies from the right columns in the label_features and year_and_month arrays
    Args:
        label_features: the np array of label features originally read in
        year_and_month: the np array of year and month, extracted from the date column

    Returns:
        all_dummies: np array of all these dummies
        dummy_names: names for all the dummy variables
    """

    # todo: with more time, make more dry, less repetitive

    type_dummies, type_feature_names = get_dummies(label_features[:, 1], "type")
    region_dummies, region_feature_names = get_dummies(label_features[:, 2], "region")

    year_dummies, year_feature_names = get_dummies(year_and_month[:, 0], "year")
    month_dummies, month_feature_names = get_dummies(year_and_month[:, 1], "month")

    all_dummies = np.concatenate(
        [type_dummies, region_dummies, year_dummies, month_dummies], axis=1
    )
    dummy_names = (
        type_feature_names
        + region_feature_names
        + year_feature_names
        + month_feature_names
    )

    return all_dummies, dummy_names


def train_test_split(X, y, train_proportion):
    """
    Random split of X (and corresponding split of y) into test and train
    Args:
        X: np array of features
        y: np array of targets
        train_proportion: proportion of training data to use for training

    Returns:
        x_train: np array of features for training
        x_test: np array of features for testing
        y_train: np array of targets for training
        y_test: np array of targets for testing
    """
    test_mask = np.random.choice(
        [False, True], len(X), p=[train_proportion, 1 - train_proportion]
    )
    train_mask = ~test_mask

    x_train = X[train_mask]
    x_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    return x_train, x_test, y_train, y_test


def initialise_coefficients(feature_names):
    """
    Construct an np array of the right shape for theta (coefficients and intercept)
    Initialise to all zeroes
    Args:
        feature_names: list of feature names (needed so we know how many elements in the coefficients)

    Returns:
        np array of the right shape for theta (coefficients and intercept)
    """
    return np.zeros((1, len(feature_names) + 1))


def add_col_of_1s_to_array(X):
    """
    Adds a column of ones to a features array. Needed so that these ones are multiplied by the intercept,
    in accordance with the framework of y = X*theta, where theta also includes the intercept
    Args:
        X: features array

    Returns:
        np array of features, with added column of ones
    """
    return np.concatenate([X, np.ones((len(X), 1))], axis=1)


def do_gradient_descent(theta, x_train, y_train, number_iterations_grad_descent):
    """
    Doing the gradient descent. Repeatedly calculate the grad of the cost wrt theta, and update theta so as to
    decrease the cost slightly. Repeat number_iterations_grad_descent times
    Args:
        theta: initial values of coefficients (np array)
        x_train: np array of training features
        y_train: np array of training targets
        number_iterations_grad_descent: number of iterations in the training process (integer)

    Returns:
        theta: final updated values of coefficients (np array)
    """
    for i in range(number_iterations_grad_descent):

        # predict on x_train
        y_hat = np.dot(theta, x_train.T)

        # get residuals and m (number of examples)
        m = y_hat.shape[1]
        residuals = y_train - y_hat

        # calculate grad of J wrt theta - direction of maximally increasing cost
        grad_J = -(1.0 / m) * np.dot(residuals, x_train)

        # calculate cost (J, the mean square error), and log the result
        # save time by doing this only every 1000 iterations
        if i % 1000 == 0:
            logger.info("Training iteration number {}".format(i))
            J = float(np.dot(residuals, residuals.T) / m)
            logger.info("MSE on train is currently {}".format(J))

        # Update theta, moving it a small amount in the direction opposite to maximally increasing cost
        # ie in the direction of maximally decreasing cost
        theta = theta - (learning_rate * grad_J)

    logger.info("Final MSE on train was {}".format(J))
    return theta


def feature_scaling(float_features):
    """
    Feature scaling - for the float variables, subtract the mean and divide by the sd.
    For faster convergence in gradient descent
    Args:
        float_features: numpy array of float features

    Returns:
        float_features_scaled: numpy array of float features, scaled to 0 mean and unit variance


    """
    means = float_features.mean(axis=0)
    means_correct_shape = np.broadcast_to(means, float_features.shape)
    sds = float_features.var(axis=0) ** 0.5
    sds_correct_shape = np.broadcast_to(sds, float_features.shape)
    float_features_scaled = (float_features - means_correct_shape) / sds_correct_shape
    return float_features_scaled


def get_equation_string(coefficients):
    """
    Combines a list of tuples of coefficients and feature names, into the linear regression model equation
    Args:
        coefficients: list of tuples of coefficients and feature names

    Returns:
        equation_string: string of linear regression model equation
    """
    equation_string = "f = " + " + ".join(
        [
            str(round(x[0], 3)) + "*" + x[1]
            if (x[1] != "intercept")
            else (str(round(x[0], 3)))
            for x in coefficients
        ]
    )
    return equation_string


def undo_feature_scaling_to_theta(means, sds, theta):
    """
    Correct for the feature scaling so that the model coefficients apply to the original (unscaled data)
    For ease of interpretation
    Args:
        means: means of unscaled float variables (np array)
        sds: standard deviations of unscaled float variables (np array)
        theta: coefficients of model fitted to scaled data

    Returns:
        theta: coefficients of model, with coefficients reflecting original scales
    """

    means = np.ndarray((1, means.shape[0]), buffer=means)
    sds = np.ndarray((1, sds.shape[0]), buffer=sds)

    means = np.concatenate(
        (means, np.zeros((1, (theta.shape[1] - means.shape[1])))), axis=1
    )
    sds = np.concatenate((sds, np.ones((1, (theta.shape[1] - sds.shape[1])))), axis=1)

    theta = np.divide((theta - means), sds)
    return theta


def sort_and_log_coefficients(theta, feature_names):
    """
    Logging the details of the resulting model
    Sort by coefficients to easily see the most important coefficients
    Args:
        theta: coefficients of model fitted to scaled data
        feature_names: list of features used in the model

    Returns:
        None
    """

    coefficients = list(zip(theta.tolist()[0], feature_names + ["intercept"]))

    coefficients = sorted(coefficients, key=itemgetter(0), reverse=True)

    equation_string = get_equation_string(coefficients)

    logger.info("Fitted model for price of avocados:")
    logger.info(equation_string)

    coefficients = np.array(coefficients)

    logger.info(
        "Coefficients for model (same numbers as in equation, presented in table format):"
    )
    logger.info(coefficients)


def evaluate_on_test(theta, x_test, y_test):
    """
    See how the model performs on test data - use to check for overfitting
    Log the mean square error of predictions on test
    Args:
        theta: model coefficients
        x_test: np array of features for testing
        y_test: np array of targets for testing

    Returns:
        None
    """
    x_test = add_col_of_1s_to_array(x_test)
    y_hat = np.dot(theta, x_test.T)
    m = y_hat.shape[1]
    residuals = y_test - y_hat
    MSE = float(np.dot(residuals, residuals.T) / m)
    logger.info("MSE on test was: {}".format(MSE))


def combine_float_and_dummy_features(
    float_features_scaled, all_dummies, float_feature_column_names, dummy_names
):
    """
    Takes the float features and the dummy variables, and combines them, into X
    Also keeps track of the feature names
    Args:
        float_features_scaled: np array of the scales float features
        all_dummies: np array of the dummy features
        float_feature_column_names: list of float feature columns
        dummy_names: list of dummy feature columns

    Returns:
        X: np array of all the features
        feature_names: list of all the feature names
    """
    X = np.concatenate((float_features_scaled, all_dummies), axis=1)
    feature_names = float_feature_column_names + dummy_names
    return X, feature_names
