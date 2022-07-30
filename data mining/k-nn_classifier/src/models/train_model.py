from numpy import loadtxt
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from numpy import savetxt


def distance_ecu(x_train, x_test_point):
    """

    Args:
        x_train:  corresponding to the training data
        x_test_point: corresponding to the test point

    Returns:
        distances: The distances between the test point and each point in the training data.

    """

    distances = []  ## create empty list called distances
    for row in range(len(x_train)):  ## Loop over the rows of x_train
        current_train_point = x_train[row]  # Get them point by point
        current_distance = 0  ## initialize the distance by zero

        for col in range(len(current_train_point)):  ## Loop over the columns of the row

            current_distance += (current_train_point[col] - x_test_point[col]) ** 2
            ## Or current_distance = current_distance + (x_train[i] - x_test_point[i])**2
        current_distance = np.sqrt(current_distance)

        distances.append(current_distance)  ## Append the distances

    # Store distances in a dataframe
    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


def nearest_neighbors(distance_point, K):
    """

    Args:
        distance_point:  the distances between the test point and each point in the training data.
        K: the number of neighbors

    Returns:
        df_nearest: the nearest K neighbors between the test point and the training data.

    """

    # Sort values using the sort_values function
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    ## Take only the first K neighbors
    df_nearest = df_nearest[:K]
    return df_nearest


def voting(df_nearest, y_train):
    """

    Args:
        df_nearest: dataframe contains the nearest K neighbors between the full training dataset and the test point.
        y_train: the labels of the training dataset.

    Returns:
        y_pred: the prediction based on Majority Voting

    """

    ## Use the Counter Object to get the labels with K nearest neighbors.
    counter_vote = Counter(y_train[df_nearest.index])
    y_pred = counter_vote.most_common()[0][0]  # Majority Voting
    return y_pred


def fit_and_store_knn_model_sklearn(K, normalized_x_train, y_train):
    """

    Args:
        K: k-neighbors
        normalized_x_train: normalized predictors to train the model.
        y_train: encoded target variable/label to be predicted during the model training.

    Returns:
        .sav file: model trained with the previous Args.

    """
    # training the model, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    # p=2 is the euclidean_distance and 'uniform' means unweighted
    knn = KNeighborsClassifier(K, weights='uniform', p=2)
    knn.fit(normalized_x_train, y_train)

    # save the model to disk
    filename = f'../../models/knn_trained_{K}_sklearn.sav'
    pickle.dump(knn, open(filename, 'wb'))


def fit_and_store_knn_model(K, normalized_x_train, y_train):
    """

    Args:
        K: k-neighbors
        normalized_x_train: normalized predictors to train the model.
        y_train: encoded target variable/label to be predicted during the model training.

    Returns:
        .sav file: model trained with the previous Args.


    """
    # training the model, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    # p=2 is the euclidean_distance and 'uniform' means unweighted
    knn = KNeighborsClassifier(K, weights='uniform', p=2)
    knn.fit(normalized_x_train, y_train)

    # save the model to disk
    filename = f'../../models/knn_trained_{K}.sav'
    pickle.dump(knn, open(filename, 'wb'))


def fit_and_predict_with_knn_model(K, normalized_x_train, y_train, normalized_x_validation):
    """

    Args:
        K: k-neighbors
        normalized_x_train: normalized predictors to train the model.
        y_train: encoded target variable/label to be predicted during the model training.
        normalized_x_validation: normalized predictors to measure the accuracy of the model.

    Returns:
        .csv file: predicted target variable/label to measure the accuracy of the model.

    """
    y_pred = []
    for x_validation_point in normalized_x_validation:
        distance_point = distance_ecu(normalized_x_train, x_validation_point)  # Step 1
        df_nearest_point = nearest_neighbors(distance_point, K)  # Step 2
        y_pred_point = voting(df_nearest_point, y_train)  # Step 3
        y_pred.append(y_pred_point)
    file_path_y_pred = f'../../models/y_pred_{K}.csv'
    savetxt(file_path_y_pred, y_pred, delimiter=',')


def main():
    """ Train model for each k-neighbors with both:
        - KNN model from sklearn
        - KNN model from scratch

    Returns:
        .sav file: model trained with the previous Args. For the KNN model from sklearn.
        .csv file: predicted target variable/label to measure the accuracy of the model. For the KNN model from scratch.

    """
    # k-neighbors to choose for training the model
    k_list = [1, 3, 5, 7, 9]

    # load necessary datasets for training the model
    file_path_normalized_x_train = '../../data/processed/normalized_x_train.csv'
    file_path_y_train = '../../data/processed/y_train.csv'

    # prediction is performed here for simplicity, for the knn without sklearn
    file_path_normalized_x_validation = '../../data/processed/normalized_x_validation.csv'
    normalized_x_validation = loadtxt(file_path_normalized_x_validation, delimiter=',')

    normalized_x_train = loadtxt(file_path_normalized_x_train, delimiter=',')
    y_train = loadtxt(file_path_y_train, delimiter=',')
    print(y_train)
    # train models based on different k-neighbors
    for k in k_list:
        fit_and_store_knn_model_sklearn(k, normalized_x_train, y_train)
        fit_and_predict_with_knn_model(k, normalized_x_train, y_train, normalized_x_validation)


if __name__ == '__main__':
    main()
