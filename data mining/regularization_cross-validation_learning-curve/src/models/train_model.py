import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import statistics

import plotly.express as px

import math


class RidgeRegression():

    def __init__(self, learning_rate, iterations, alpha):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.alpha = alpha

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros(self.n)

        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X)

        # calculate gradients
        dW = (- (2 * (self.X.T).dot(self.Y - Y_pred)) +
              (2 * self.alpha * self.W)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b


def truncate(number, decimals=0):
    """     Returns a value truncated to a specific number of decimal places.

    Args:
        number: number to format.
        decimals: decimals to show.

    Returns:
        float number: with the number of specified decimals.

    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


def x_y_split(df):
    """

    Args:
        df:  dataset

    Returns:
        x: array with the predictors
        y: array with the target to predict

    """
    x = np.asarray(df.iloc[:, :-1])  # predictors
    y = np.asarray(df.iloc[:, -1])  # target
    # print("Length of x: ", len(x), 'Length of y:', len(y))

    return x, y


def read_csv_from_folder(folder_name, file_name):
    """

    Args:
        folder_name: file
        file_name: file

    Returns:
        df_train: data frame with training data from the files.

    """
    file_path_train = '../../data' + '/' + folder_name + '/' + file_name + '.csv'
    df_train = pd.read_csv(file_path_train)
    return df_train


def plot_mse_vs_lambda(df_mse_lambdas, columns, file_train, file_test):
    """

    Args:
        df_mse_lambdas: dataframe with information about mse and lambdas
        columns: column names
        file_train: file to get the training data
        file_test: file to get the testing data

    Returns:
        plot: plots the MSE vs Lambda

    """
    fig = px.scatter(df_mse_lambdas, x=columns[2], y=df_mse_lambdas.columns[0:2],
                     # trendline="ols",
                     labels={
                         columns[2]: "Lambda"
                     },
                     title=f"MSE vs Lambda for {file_train} and {file_test} ")

    fig.update_layout(
        yaxis_title="Mean Square Error"
    )

    fig.show()


def build_df_to_plot(mse_train, mse_test, lambdas):
    """

    Args:
        mse_train: list of MSE measured using the training data.
        mse_test: list pf MSE measured using the testing data.
        lambdas: list of lambdas

    Returns:
        df_mse_lambdas: dataframe with MSE and lambdas.
        columns: name of the columns of the df_mse_lambdas.

    """
    columns = ['Mean Square Error Test', 'Mean Square Error Train', 'Lambdas']
    df_mse_lambdas = pd.DataFrame({columns[0]: mse_test,
                                   columns[1]: mse_train,
                                   columns[2]: lambdas
                                   })
    return df_mse_lambdas, columns


def display_min_mse_lambda(df_mse_lambdas, columns, file_train, file_test):
    """

    Args:
        df_mse_lambdas: dataframe with MSE for each lambda
        columns: column names
        file_train: name of the training data set.
        file_test: name of the testing data set.

    Returns:
        min_error_df: dataframe with the corresponding minimum MSE and lambda.
        file_train: name of the training data set.
        file_test: name of the testing data set.

    """
    min_error_df = df_mse_lambdas.loc[df_mse_lambdas[columns[0]] == df_mse_lambdas[columns[0]].min()].iloc[:,
                   [0, 2]]
    min_error = min_error_df.to_string(index=False)
    print("The Lambda with less square error using the training set ",
          file_train, " and the testing set ", file_test, "is: \n", min_error)
    return min_error_df, file_train, file_test


def display_min_mse_lambda_k_fold(df_mse_lambdas, columns, file_train):
    """

    Args:
        df_mse_lambdas: dataframe with MSE for each lambda
        columns: column names
        file_train: name of the training data set.

    Returns:
        min_error_df: dataframe with the corresponding minimum MSE and lambda.

    """
    min_error_df = df_mse_lambdas.loc[df_mse_lambdas[columns[0]] == df_mse_lambdas[columns[0]].min()].iloc[:,
                   [0, 1]]
    print("The Lambda with less square error using the training set ",
          file_train, " is: \n", min_error_df.to_string(index=False))
    return min_error_df


def regularized_model_for_train_test(x_train, y_train, x_test, y_test, first_lambda, ridge_scratch):
    """

    Args:
        x_train:
        y_train:
        x_test:
        y_test:
        first_lambda: lambda to start iteration from.

    Returns:
        mse_train:
        mse_test:
        lambdas:

    """
    # linear_regression_model = linear_model.LinearRegression()
    mse_test = []
    mse_train = []
    lambdas = []
    for lambda_value in range(first_lambda, 150 + 1):
        if ridge_scratch == True:
            regularized_linear_regression_model = RidgeRegression(iterations=1000,
                                                                  learning_rate=0.01,
                                                                  alpha=lambda_value)
        else:
            regularized_linear_regression_model = Ridge(alpha=lambda_value)

        regularized_linear_regression_model.fit(x_train, y_train)

        # Make predictions using the testing set
        y_pred_test = regularized_linear_regression_model.predict(x_test)
        mse_test.append(mean_squared_error(y_test, y_pred_test))
        # Make predictions using the training set
        y_pred_train = regularized_linear_regression_model.predict(x_train)
        mse_train.append(mean_squared_error(y_train, y_pred_train))

        lambdas.append(lambda_value)
    return mse_train, mse_test, lambdas


def k_fold_cv(X, y, splits, lambda_value, ridge_scratch):
    """

    Args:
        X: training data set.
        y: testing data set.
        splits: number of folds.
        lambda_value: int with the value of lambda.

    Returns:
        average_mse: float

    """
    k_fold = KFold(n_splits=splits, shuffle=True, random_state=100)
    mse_each_fold = []
    count = 0
    for train_index, test_index in k_fold.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        if ridge_scratch == True:
            regularized_linear_regression_model = RidgeRegression(iterations=1000,
                                                                  learning_rate=0.01,
                                                                  alpha=lambda_value)
        else:
            regularized_linear_regression_model = Ridge(alpha=lambda_value)

        regularized_linear_regression_model.fit(X_train, y_train)
        y_pred_train = regularized_linear_regression_model.predict(X_test)
        mse_fold = mean_squared_error(y_test, y_pred_train)
        count += 1
        # print("Fold number: ", count, ", MSE: ", mse_fold, ", Lambda: ", lambda_value)
        mse_each_fold.append(mse_fold)
    average_mse = statistics.mean(mse_each_fold)
    # print(f"Average MSE for lambda {lambda_value}: ", average_mse)

    return average_mse


def train_test_split_custom(df_train, df_test):
    """

    Args:
        df_train:data frame with training data from the files.
        df_test: data frame with the testing data.

    Returns:
        x_train: array with the predictors from the training data.
        y_train: array with the target to predict from the training data.
        x_test: array with the predictors from the testing data.
        y_test: array with the target to predict from the testing data.


    """
    # Preparing the 'training data' predictors ('x') and target ('y') to train the model.
    x_train, y_train = x_y_split(df_train)
    # print("Number of samples of the training set: ", x_train.shape[0])
    # print("Number of features of the training set: ", x_train.shape[1])

    x_test, y_test = x_y_split(df_test)
    # print("Number of samples of the testing set: ", x_test.shape[0])
    # print("Number of features of the testing set: ", x_test.shape[1])
    return x_train, y_train, x_test, y_test


def df_train_test(folder, file_train, file_test):
    """

    Args:
        folder: folder containing the files.
        file_train: training data set.
        file_test: testing data set.

    Returns:
        df_train: dataframe with the training data.
        df_test: dataframe with the testing data.

    """
    df_train = read_csv_from_folder(folder, file_train)
    df_test = read_csv_from_folder(folder, file_test)
    return df_train, df_test


def regularization_linear_regression_lambda_N_150(train_test_files, first_lambda, ridge_scratch):
    """

    Args:
        train_test_files: dictionary with the names of the training and test files.
        first_lambda: int, first lambda to be used.
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        df_min_mse_lambda: dataframe with the Minimum MSE and its lambda.

    """
    folder = 'raw'
    list_datasets_analyzed = []
    for file_train, file_test in train_test_files.items():
        print('Regularized lineal models for:', file_train, ' and ', file_test)
        df_train, df_test = df_train_test(folder, file_train, file_test)
        x_train, y_train, x_test, y_test = train_test_split_custom(df_train, df_test)
        mse_train, mse_test, lambdas = regularized_model_for_train_test(x_train, y_train, x_test, y_test,
                                                                        first_lambda, ridge_scratch)
        df_mse_lambdas, columns = build_df_to_plot(mse_train, mse_test, lambdas)

        min_error, file_train, file_test = display_min_mse_lambda(df_mse_lambdas, columns, file_train, file_test)
        # print("ERROR", min_error.iloc[0,0])
        # print("LAMBDA", min_error.iloc[0,1])

        list_datasets_analyzed.append([file_train,
                                       file_test,
                                       min_error.iloc[0, 0],
                                       min_error.iloc[0, 1]
                                       ])
        plot_mse_vs_lambda(df_mse_lambdas, columns, file_train, file_test)

    columns = ['file_train', 'file_test',
               'Min Mean Square Error Test', 'Lambda Min MSE']
    df_min_mse_lambda = build_df_from_list(list_datasets_analyzed, columns)
    print("Current lineal models finished and analyzed.\n\n")
    return df_min_mse_lambda


def lambda_selection_cv(x_train, y_train, n_splits, file_train, lambda_range, ridge_scratch):
    """

    Args:
        x_train: array with the predictors of the training data.
        y_train: array with the target to predict of the training data.
        n_splits: int, number of splits
        file_train: training file.
        lambda_range: list containing the min and max value of the range.
        ridge_scratch: bool, ridge regression from sklearn or from scratch


    Returns:
        lambda_min_mse: dataframe with the lambda and its minimum MSE.

    """
    lambda_values = []
    average_mses = []
    for lambda_value in range(lambda_range[0], lambda_range[1] + 1):
        lambda_values.append(lambda_value)
        average_mses.append(k_fold_cv(x_train, y_train, n_splits, lambda_value, ridge_scratch))

    columns = ['Average MSE', 'Lambda']
    df_mse_lambdas = pd.DataFrame({columns[0]: average_mses,
                                   columns[1]: lambda_values})
    lambda_min_mse = display_min_mse_lambda_k_fold(df_mse_lambdas, columns, file_train).iloc[0, 1]
    return lambda_min_mse


def regularized_lin_reg_model_best_lambda(df_train, df_test, lambda_min_mse, ridge_scratch):
    """

    Args:
        df_train:data frame with training data from the files.
        df_test: data frame with the testing data.
        lambda_min_mse: lambda with the minimum MSE
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        mean_squared_error: calculated from the predictions using test file.

    """
    x_train, y_train, x_test, y_test = train_test_split_custom(df_train, df_test)
    if ridge_scratch == True:
        regularized_linear_regression_model = RidgeRegression(iterations=1000,
                                                              learning_rate=0.01,
                                                              alpha=lambda_min_mse)
    else:
        regularized_linear_regression_model = Ridge(alpha=lambda_min_mse)

    regularized_linear_regression_model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred_test = regularized_linear_regression_model.predict(x_test)
    return mean_squared_error(y_test, y_pred_test)


def lambda_selection_cv_mse_test(train_test_files, n_splits, lambda_range, ridge_scratch):
    """

    Args:
        train_test_files: training and test files.
        n_splits: number of splits.
        lambda_range: list with the first and last value of lambdas.
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        list_df_mse_cv_lambdas: list with the MSE for a specific lambda and train-test files.

    """
    folder = 'raw'
    count_dataset = 0
    list_df_mse_cv_lambdas = []
    for file_train, file_test in train_test_files.items():
        print("Data set number:", count_dataset + 1, "\n")
        print('Regularized lineal models for:', file_train, ' and ', file_test)
        df_train, df_test = df_train_test(folder, file_train, file_test)
        x_train, y_train = x_y_split(df_train)
        lambda_min_mse = lambda_selection_cv(x_train, y_train, n_splits, file_train, lambda_range, ridge_scratch)

        mse_test_set = regularized_lin_reg_model_best_lambda(df_train, df_test, lambda_min_mse, ridge_scratch)
        print("\nResults with the optimal lambda value based on cross-validation: \n")
        print("Training set: ",
              file_train, "\nTesting set: ", file_test, "\nMSE: ",
              truncate(mse_test_set, decimals=2), "\nLambda: ",
              lambda_min_mse)
        list_df_mse_cv_lambdas.append([file_train, file_test, mse_test_set, lambda_min_mse])
        count_dataset += 1
        print('\n')
    return list_df_mse_cv_lambdas


def get_six_datasets():
    """

    Returns:
        train_test_files: dictionary with the names of pair files, train and test.

    """
    train_test_files = dict({'train-50(1000)-100': 'test-1000-100',
                             'train-100(1000)-100': 'test-1000-100',
                             'train-150(1000)-100': 'test-1000-100',
                             'train-100-10': 'test-100-10',
                             'train-100-100': 'test-100-100',
                             'train-1000-100': 'test-1000-100'
                             })
    return train_test_files


def get_additional_files():
    """

    Returns:
        train_test_files: dictionary with the names of pair files, train and test.


    """
    train_test_files = dict({'train-100-100': 'test-1000-100',
                             'train-50(1000)-100': 'test-1000-100',
                             'train-100(1000)-100': 'test-1000-100'})
    return train_test_files


def plot_mse_train_validation_vs_training_size(df_mse_training_size, columns, file_train):
    """

    Args:
        df_mse_training_size: dataframe with MSE and training size info.
        columns: names of the columns.
        file_train: training file.

    Returns:
        plot: MSE vs training size.


    """
    fig1 = px.line(df_mse_training_size, x=columns[2], y=df_mse_training_size.columns[0:2],
                   labels={
                       columns[2]: "Training Size %"
                   },
                   title=f"MSE vs Training size for {columns[3]} {df_mse_training_size.iloc[1, 3]} and file"
                         f" {file_train}")

    fig1.update_layout(
        yaxis_title="Mean Square Error",
    )

    fig1.show()


def train_test_split_from_scratch(df_train, test_size):
    """

    Args:
        df_train: dataframe with the training data.
        test_size:

    Returns:
        x_train:
        y_train:
        x_validation:
        y_validation:

    """
    mask = np.random.rand(len(df_train)) <= test_size
    training_data = df_train[~mask]
    testing_data = df_train[mask]
    x_train, y_train = x_y_split(training_data)
    x_validation, y_validation = x_y_split(testing_data)
    return x_train, y_train, x_validation, y_validation


def get_mse_test_train_vs_training_size(list_test_size, df_train, iterations, lambda_value, ridge_scratch):
    """

    Args:
        list_test_size: list with values of the percentage of training size for the ridge regression.
        df_train: dataframe, training data.
        lambda_value: int, value of lambda.
        file_train: name of the training file.
        iterations: number of iterations to average the MSE.
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        mse_validation: float with MSE.
        mse_train: float with MSE.
        training_size: % of training size.


    """
    mse_train = []
    mse_validation = []
    training_size = []
    for test_size in list_test_size:
        mse_train_process = []
        mse_validation_process = []
        # for a smoother plot
        for process in range(iterations):
            x_train, y_train, x_validation, y_validation = train_test_split_from_scratch(df_train, test_size)
            if ridge_scratch == True:
                regularized_linear_regression_model = RidgeRegression(iterations=1000,
                                                                      learning_rate=0.01,
                                                                      alpha=lambda_value)
            else:
                regularized_linear_regression_model = Ridge(alpha=lambda_value)

            regularized_linear_regression_model.fit(x_train, y_train)

            # Make predictions using the train set
            y_pred_train = regularized_linear_regression_model.predict(x_train)
            mse_train_process.append(mean_squared_error(y_train, y_pred_train))

            # Make predictions using the validation set
            y_pred_validation = regularized_linear_regression_model.predict(x_validation)
            mse_validation_process.append(mean_squared_error(y_validation, y_pred_validation))

        mse_train.append(statistics.mean(mse_train_process))
        mse_validation.append(statistics.mean(mse_validation_process))
        training_size.append((1 - test_size) * 100)
    return mse_validation, mse_train, training_size


def calculate_mse_train_validation_vs_training_size(df_train, lambda_value, file_train, step, iterations,
                                                    ridge_scratch):
    """

    Args:
        df_train: dataframe, training data.
        lambda_value: int, value of lambda.
        file_train: name of the training file.
        step: increase of the training size for each iteration.
        iterations: number of iterations to average the MSE.
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        plot mse_train_validation_vs_training_size

    """
    list_test_size = list(np.arange(0.01, 0.95, step))
    list_test_size = [truncate(item, decimals=4) for item in list_test_size]
    mse_validation, mse_train, training_size = get_mse_test_train_vs_training_size(list_test_size, df_train, iterations,
                                                                                   lambda_value, ridge_scratch)
    columns = ['Mean Square Error Validation', 'Mean Square Error Train', 'Training size', 'Lambda']
    df_mse_training_size = pd.DataFrame({columns[0]: mse_validation,
                                         columns[1]: mse_train,
                                         columns[2]: training_size,
                                         columns[3]: lambda_value
                                         })
    plot_mse_train_validation_vs_training_size(df_mse_training_size, columns, file_train)


def analyze_all_datasets(ridge_scratch):
    """

    Args:
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns: analysis of all datasets with lambda from 0 to 150.

    """
    first_lambda = 0
    train_test_files = get_six_datasets()
    df_min_mse_lambda = regularization_linear_regression_lambda_N_150(train_test_files, first_lambda, ridge_scratch)
    return df_min_mse_lambda


def analyze_additional_datasets(ridge_scratch):
    """

    Args:
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns: analysis of additional datasets, with lambda from 1 to 150.

    """
    train_test_files = get_additional_files()
    first_lambda = 1
    print("\n Additional graph with λ ranging from 1 to 150.\n")
    regularization_linear_regression_lambda_N_150(train_test_files, first_lambda, ridge_scratch)


def regularization_2(ridge_scratch):
    """
    Args:
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        graphs: MSE for different lambdas and training and test files.

    """
    min_mse_lambda_df = analyze_all_datasets(ridge_scratch)
    analyze_additional_datasets(ridge_scratch)
    return min_mse_lambda_df


def build_df_from_list(list_values, columns):
    """

    Args:
        list_values:
        columns:

    Returns:

    """
    df = pd.DataFrame(list_values, columns=columns)
    return df


def calculate_mse_lambda_diff(df_differences):
    """

    Args:
        df_differences: dataframe to calculate differences from.

    Returns:
        df_differences: dataframe with differences.

    """
    new_columns = ['Diff MSE (Optimal CV- Min)', 'Diff Lambdas (Optimal CV - Min)']
    df_differences[new_columns[0]] = df_differences['Optimal Mean Square Error Test CV'] - \
                                     df_differences['Min Mean Square Error Test']

    df_differences[new_columns[1]] = \
        df_differences['Optimal Lambda CV'] - df_differences[
            'Lambda Min MSE']

    df_differences.sort_values(by=new_columns[0],
                               ascending=True,
                               inplace=True)

    return df_differences


def build_df_differences(min_mse_lambda_df, list_df_mse_optimal_lambdas):
    """

    Args:
        min_mse_lambda_df:
        list_df_mse_optimal_lambdas:

    Returns:
        df_differences: dataframe with differences.

    """
    columns = ['file_train', 'file_test',
               'Optimal Mean Square Error Test CV',
               'Optimal Lambda CV']
    df_mse_optimal_lambdas_cv = build_df_from_list(list_df_mse_optimal_lambdas, columns)

    columns = ['file_train', 'file_test']
    df_merged = min_mse_lambda_df.merge(df_mse_optimal_lambdas_cv, how='inner',
                                        on=columns)
    df_differences = calculate_mse_lambda_diff(df_merged)
    return df_differences


def df_differences_to_csv(df_differences):
    """

    Args:
        df_differences: dataframe with the differences between MSE and lambdas

    Returns:
        df_differences_optimal_vs_min_mse_lambda.csv: file with the differences between MSE and lambdas

    """
    folder_name = 'output'
    file_name = 'df_differences_optimal_vs_min_mse_lambda'
    file_path = '../../data' + '/' + folder_name + '/' + file_name + '.csv'
    df_differences_optimal_vs_min_mse_lambda = df_differences.copy()
    df_differences_optimal_vs_min_mse_lambda.to_csv(file_path, index=False)


def calculate_differences_mse_lambdas(min_mse_lambda_df, list_df_mse_optimal_lambdas):
    """

    Args:
        min_mse_lambda_df: dataframe with info about min MSE and its lambdas.
        list_df_mse_optimal_lambdas: list of optimal lambdas with min MSE from CV

    Returns:
        df_differences_optimal_vs_min_mse_lambda.csv: file with the differences between MSE and lambdas


    """
    df_differences = build_df_differences(min_mse_lambda_df, list_df_mse_optimal_lambdas)
    df_differences_to_csv(df_differences)


def cross_validation_3(min_mse_lambda_df, ridge_scratch):
    """
    Args:
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        Summary of the MSE for a specific lambda and train-test files. Selecting first the best lambda using
        cross-validation.


    """
    train_test_files = get_six_datasets()
    n_splits = 10
    lambda_range = [0, 150]
    list_df_mse_cv_lambdas = lambda_selection_cv_mse_test(train_test_files, n_splits, lambda_range, ridge_scratch)
    calculate_differences_mse_lambdas(min_mse_lambda_df, list_df_mse_cv_lambdas)


def learning_curve_4(ridge_scratch):
    """
    Args:
        ridge_scratch: bool, ridge regression from sklearn or from scratch

    Returns:
        plot: with the learning curve (MSE and training size) for different lambdas.

    """
    train_test_files = dict({'train-1000-100': 'test-1000-100'})
    folder = 'raw'

    lambdas = [1, 25, 150]
    step = 0.01  # 0.01
    iterations = 100  # 100

    for file_train, file_test in train_test_files.items():
        # df_test will not be used
        df_train, df_test = df_train_test(folder, file_train, file_test)
        for lambda_value in lambdas:
            calculate_mse_train_validation_vs_training_size(df_train, lambda_value, file_train, step, iterations,
                                                            ridge_scratch)


def main():
    """

    Returns: Results from studying multiple training and testing sets, with different lambdas, to apply regularization,
    using cross-validation to select the best lambda for the regularization, and different training sizes. Measuring
    the MSE for all of them.

    """
    # Ridge Regression from scratch (True) or from sklearn (False)
    ridge_from_scratch = False
    min_mse_lambda_df = regularization_2(ridge_from_scratch)
    cross_validation_3(min_mse_lambda_df, ridge_from_scratch)
    learning_curve_4(ridge_from_scratch)

    print("Process done.")


if __name__ == '__main__':
    main()
