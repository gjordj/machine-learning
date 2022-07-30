from train_model import *
from numpy import loadtxt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def build_output(x_data, output):
    """ Build output with the format requested for the prediction done with the validation data.


    Args:
        x_data: predictors to fill the dataframe.
        output: prediction data to fill the dataframe.

    Returns:
        output: dataframe with predictors data and predictions for each k-neighbors.

    """
    plant_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df_plant_features = pd.DataFrame(x_data, columns=plant_features)
    output = pd.concat([df_plant_features, output], axis=1, join="inner")
    return output


def calculate_accuracy_and_build_output(label_encoder, normalized_x_validation, y_validation,
                                        k_list=[1, 3, 5, 7, 9]):
    """

    Args:
        label_encoder: to decode the target variable y into its original meaning.
        normalized_x_validation: validation data to perform predictions with and measure accuracy with
        y_validation: validation target to measure accuracy with
        k_list: k-neighbors

    Returns:
        output: dataframe with the predictions based on each out of sample.

    """

    # Data to build the output in the requested format
    file_path_x_validation = '../../data/processed/x_validation.csv'
    x_validation = loadtxt(file_path_x_validation, delimiter=',')
    target_names = ['setosa', 'versicolor', 'virginica']

    to_build = pd.DataFrame(columns=k_list)
    for k in k_list:
        file_path_y_pred = f'../../models/y_pred_{k}.csv'
        y_pred = loadtxt(file_path_y_pred, delimiter=',')

        # metrics, comparing the result of the prediction with the y_validation
        print(f'The performance metrics of the knn-classifier built from scratch with {k} neighbors are:\n')
        print(classification_report(y_validation, y_pred, target_names=target_names))

        y_pred_knn_classifier = pd.Series(y_pred).astype(int)
        prediction_by_predictors = label_encoder.inverse_transform(y_pred_knn_classifier)
        to_build[k] = prediction_by_predictors
        # build output with the format requested for the prediction done with the validation data.
        output = build_output(x_validation, to_build)
    print('\n')
    return output


def predict_calculate_accuracy_and_build_output(label_encoder, normalized_x_validation, y_validation,
                                                k_list=[1, 3, 5, 7, 9]):
    """

    Args:
        label_encoder: to decode the target variable y into its original meaning.
        normalized_x_validation: validation data to perform predictions with and measure accuracy with
        y_validation: validation target to measure accuracy with
        k_list: k-neighbors

    Returns:
        output: dataframe with the predictions based on each out of sample.

    """

    # Data to build the output in the requested format
    file_path_x_validation = '../../data/processed/x_validation.csv'
    x_validation = loadtxt(file_path_x_validation, delimiter=',')
    target_names = ['setosa', 'versicolor', 'virginica']

    to_build = pd.DataFrame(columns=k_list)
    for k in k_list:
        # load the model from disk
        filename = f'../../models/knn_trained_{k}_sklearn.sav'
        loaded_knn_model = pickle.load(open(filename, 'rb'))
        # predict based on the predictors
        y_pred_knn_classifier = loaded_knn_model.predict(normalized_x_validation)
        # accuracy, comparing the result of the prediction with the y_validation
        print(f'The performance metrics of the knn-classifier built from scratch with {k} neighbors are:\n')
        print(classification_report(y_validation, y_pred_knn_classifier, target_names=target_names))

        y_pred_knn_classifier = pd.Series(y_pred_knn_classifier).astype(int)
        prediction_by_predictors = label_encoder.inverse_transform(y_pred_knn_classifier)
        to_build[k] = prediction_by_predictors
        # build output with the format requested for the prediction done with the validation data.
        output = build_output(x_validation, to_build)
    print('\n')
    return output


def predict_and_build_output_with_test_data(label_encoder, normalized_x_test, k_list=[1, 3, 5, 7, 9]):
    """

    Args:
        label_encoder: to decode the target variable y into its original meaning.
        normalized_x_test: test data out of sample to perform predictions with
        k_list: k-neighbors

    Returns:
        output: dataframe with the predictions based on each out of sample.

    """

    # Data to build the output in the requested format
    file_path_test = '../../data/interim/test_df.pkl'
    x_test = pd.read_pickle(file_path_test)  # it doesn't have the Label column.
    x_test = x_test.to_numpy()

    to_build = pd.DataFrame(columns=k_list)
    for k in k_list:
        # load the model from disk
        filename = f'../../models/knn_trained_{k}_sklearn.sav'
        loaded_knn_model = pickle.load(open(filename, 'rb'))
        # predict based on the predictors
        y_pred_knn_classifier = loaded_knn_model.predict(normalized_x_test)
        # build output
        y_pred_knn_classifier = pd.Series(y_pred_knn_classifier).astype(int)
        prediction_by_predictors = label_encoder.inverse_transform(y_pred_knn_classifier)
        to_build[k] = prediction_by_predictors
        output = build_output(x_test, to_build)
    return output


def main():
    """ Measure the accuracy of the model with the validation data. For the 2 models:
        - KNN from scratch.
        - KNN from sklearn.
        Make the predictions with the test data, and build a .csv file with the requested and decoded format of the
        target variable, taking into account each k-neighbors.

    Returns:
        output.csv: file with the x predictors and predicted y / label, for each k-neighbors, with the requested format.

    """
    # k-neighbors to choose for predicting the model
    k_list = [1, 3, 5, 7, 9]

    # load validation data, predictors x_validation and target variable y_validation
    file_path_normalized_x_validation = '../../data/processed/normalized_x_validation.csv'
    file_path_normalized_x_test = '../../data/processed/normalized_x_test.csv'
    file_path_y_validation = '../../data/processed/y_validation.csv'

    # load properly formatted validation data to measure accuracy later on
    normalized_x_validation = loadtxt(file_path_normalized_x_validation, delimiter=',')
    y_validation = loadtxt(file_path_y_validation, delimiter=',')

    # load properly formatted test data to predict with new out of sample data
    normalized_x_test = loadtxt(file_path_normalized_x_test, delimiter=',')

    # load encoder to decode categorical variables from the prediction later on
    file_encoder_path = '../../data/interim/label_encoder.npy'
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(file_encoder_path, allow_pickle=True)

    # accuracy of the model based on x and y validation data. With the KNN using sklearn
    predict_calculate_accuracy_and_build_output(label_encoder, normalized_x_validation,
                                                y_validation, k_list)

    # accuracy of the model based on x and y validation data. With the KNN from scratch
    calculate_accuracy_and_build_output(label_encoder, normalized_x_validation, y_validation,
                                        k_list)

    # predict and build output with the format requested for the test data, using the previously trained models
    output = predict_and_build_output_with_test_data(label_encoder, normalized_x_test,
                                                     k_list)
    print("The results of the predictions with the test data (out of sample) are:\n", output)

    # output file in .csv format with the requested format
    file_path_output = '../../data/output/output.csv'
    output.to_csv(file_path_output, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()
