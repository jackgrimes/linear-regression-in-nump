from configs import (
    float_feature_column_names,
    col_index_of_date,
    train_proportion,
    number_iterations_grad_descent,
)
from utils import (
    read_data,
    extract_year_and_month,
    get_all_dummies,
    train_test_split,
    initialise_coefficients,
    add_col_of_1s_to_array,
    do_gradient_descent,
    feature_scaling,
    sort_and_log_coefficients,
    evaluate_on_test,
    combine_float_and_dummy_features,
)


def runner():
    """
    Overall runner function for fitting the linear regression model to the data
    Returns: None
    """
    y, float_features, label_features = read_data()

    float_features_scaled = feature_scaling(float_features)

    year_and_month = extract_year_and_month(
        label_features, col_index_of_date=col_index_of_date
    )

    all_dummies, dummy_names = get_all_dummies(label_features, year_and_month)

    X, feature_names = combine_float_and_dummy_features(
        float_features_scaled, all_dummies, float_feature_column_names, dummy_names
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_proportion)

    x_train = add_col_of_1s_to_array(x_train)

    theta = initialise_coefficients(feature_names)

    theta = do_gradient_descent(theta, x_train, y_train, number_iterations_grad_descent)

    sort_and_log_coefficients(theta, feature_names)

    evaluate_on_test(theta, x_test, y_test)


if __name__ == "__main__":
    runner()
