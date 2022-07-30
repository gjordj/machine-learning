import pandas as pd
import numpy as np
from numpy import savetxt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer


def integer_encode_categorical_column(df, column):
    """

    Args:
        df: dataframe containing the target  / label column to be encoded.
        column: column to be encoded

    Returns:
        df: dataframe with the encoded label column.
        encoding_reference: dataframe showing how each label has been encoded.


    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df[column])

    # References of the encoding
    encoding_reference = df.copy()
    encoding_reference['encoded'] = label_encoder.fit_transform(df[column].values)
    encoding_reference = encoding_reference.drop_duplicates(column)
    encoding_reference = encoding_reference.sort_values('encoded').reset_index().iloc[:, -2:]

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    df[column] = integer_encoded
    file_encoder_path = '../../data/interim/label_encoder.npy'
    np.save(file_encoder_path, label_encoder.classes_)

    return df, encoding_reference


def x_y_split(df):
    """

    Args:
        df: training dataset

    Returns:
        x: predictors of the training dataset.
        y: target to predict of the training dataset.

    """
    x = df.iloc[:, :-1]  # predictors
    y = df.iloc[:, -1]  # target
    # print('Predictors X: \n', x.head())
    # print('Target to predict Y: \n', y.head())
    return x, y


def train_test_split_formatted_as_array(x, y):
    """ Shuffle the training data and split it into a:
    - training set: made of:
        - x predictors
        - y target to predict
     - test set: made of:
        - x predictors
        - y target to predict

    Args:
        x: predictors of the training dataset.
        y: target to predict of the training dataset.

    Returns:
        x_train: predictors to train the model and predict y_train.
        y_train: target to predict to train the model.

        x_validation: predictors to test the accuracy of the model and predict y_validation.
        y_validation: target to predict to test the accuracy of the model.

    """
    x_train, x_validation, y_train, y_validation = train_test_split(x, y,
                                                                    test_size=0.2,
                                                                    shuffle=True,  # shuffle the data to avoid bias
                                                                    random_state=0)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_validation = np.asarray(x_validation)
    y_validation = np.asarray(y_validation)
    print(f'Training set size: {x_train.shape[0]} samples.')
    print(f'Validation set size: {x_validation.shape[0]} samples')

    return x_train, x_validation, y_train, y_validation


def normalize_df(x_train, x_validation, x_test):
    """
    Note: Note how the conversion of x_train using a function which fits (figures out the params) then normalises.
    Whilst the x_validation conversion just transforms, using the same params that it learned from the train data.

    Args:
        x_train: training predictors
        x_validation: test predictors

    Returns:
        normalized_x_train: normalized x_train
        normalized_x_validation: normalized x_validation


    """
    scaler = Normalizer().fit(x_train)  # the scaler is fitted to the training set
    normalized_x_train = scaler.transform(x_train)  # the scaler is applied to the training set
    normalized_x_validation = scaler.transform(x_validation)  # the scaler is applied to the validation set
    normalized_x_test = scaler.transform(x_test)

    return normalized_x_train, normalized_x_validation, normalized_x_test


def main():
    """

       Returns: the processed data sets, by:
           - splitting them into:
               - train, validation sets
               - x (predictors) and y (target variable, encoded)

           - normalizing and formatting:
                - train, validation and test sets,
                - to properly train the KNN Classifier later on.

       """

    # Read training df
    file_path_train = '../../data/interim/train_df.pkl'
    file_path_test = '../../data/interim/test_df.pkl'

    # train data to split into train and validation set
    df_train = pd.read_pickle(file_path_train)

    # test data, out of sample
    x_test = pd.read_pickle(file_path_test)  # it doesn't have the Label column.
    x_test = np.asarray(x_test)  # format to train the model later on

    # encode categorical column (our target) from the training dataset
    df_train, df_encoding_reference = integer_encode_categorical_column(df_train, df_train.columns[-1])

    # Preparing the 'training data' predictors ('x') and target ('y') to train the model.
    x, y = x_y_split(df_train)
    x_train, x_validation, y_train, y_validation = train_test_split_formatted_as_array(x, y)

    # Normalize the Dataset: to avoid that when one feature values are larger than other, that feature dominates
    # the distance needed in the KNN algorithm.
    normalized_x_train, normalized_x_validation, normalized_x_test = normalize_df(x_train, x_validation, x_test)

    # storing processed data in the processed folder, ready for modeling
    file_path_x_train = '../../data/processed/x_train.csv'
    file_path_x_validation = '../../data/processed/x_validation.csv'

    file_path_normalized_x_train = '../../data/processed/normalized_x_train.csv'
    file_path_normalized_x_validation = '../../data/processed/normalized_x_validation.csv'
    file_path_normalized_x_test = '../../data/processed/normalized_x_test.csv'

    file_path_y_train = '../../data/processed/y_train.csv'
    file_path_y_validation = '../../data/processed/y_validation.csv'

    savetxt(file_path_x_train, x_train, delimiter=',')
    savetxt(file_path_x_validation, x_validation, delimiter=',')

    savetxt(file_path_normalized_x_train, normalized_x_train, delimiter=',')
    savetxt(file_path_normalized_x_validation, normalized_x_validation, delimiter=',')
    savetxt(file_path_normalized_x_test, normalized_x_test, delimiter=',')

    savetxt(file_path_y_train, y_train, delimiter=',')
    savetxt(file_path_y_validation, y_validation, delimiter=',')

    print("The different classes have been encoded this way:\n", df_encoding_reference)


if __name__ == '__main__':
    main()
