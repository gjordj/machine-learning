import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from numpy import savetxt
import numpy as np
import pickle
import matplotlib.pyplot as plt
import collections
from scipy.io import arff
import scipy.stats

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


def get_pie_chart_values(target_variable):
    counter = collections.Counter(target_variable)
    labels = ['Low Salary', 'High Salary']
    sizes = [counter['Low'], counter['High']]
    return labels, sizes


def pie_chart(target_variable):
    labels, sizes = get_pie_chart_values(target_variable)
    colors = ['#66b3ff', '#ffcc99']
    fig1, ax1 = plt.subplots()
    patches, texts, autotexts = ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
    for text in texts:
        text.set_color('grey')
    for autotext in autotexts:
        autotext.set_color('grey')
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()


def ordinal_encode(df, categories, column):
    encoder = OrdinalEncoder(categories=[categories],
                             handle_unknown='use_encoded_value', unknown_value=np.nan)
    df[df.columns[column]] = encoder.fit_transform(df.loc[:, [df.columns[column]]])
    return df[df.columns[column]]


def store_sets(X_train, X_test, y_train, y_test, columns):
    file_path_X_train = '../../data/processed/X_train.csv'
    file_path_y_train = '../../data/processed/y_train.csv'
    file_path_X_test = '../../data/processed/X_test.csv'
    file_path_y_test = '../../data/processed/y_test.csv'
    file_path_columns_names = '../../data/processed/columns_names.pickle'

    savetxt(file_path_X_train, X_train, delimiter=',')
    savetxt(file_path_y_train, y_train, delimiter=',')
    savetxt(file_path_X_test, X_test, delimiter=',')
    savetxt(file_path_y_test, y_test, delimiter=',')

    columns = columns[:-1]
    # print("Features Names", columns)
    open_file = open(file_path_columns_names, "wb")
    pickle.dump(columns, open_file)
    open_file.close()


def store_vehicle_sets(X, Y, X_train, X_test, y_train, y_test):
    file_path_X = '../../data/interim/vehicle/X.csv'
    file_path_y = '../../data/interim/vehicle/Y.csv'
    file_path_X_train = '../../data/interim/vehicle/X_train.csv'
    file_path_y_train = '../../data/interim/vehicle/y_train.csv'
    file_path_X_test = '../../data/interim/vehicle/X_test.csv'
    file_path_y_test = '../../data/interim/vehicle/y_test.csv'

    savetxt(file_path_X, X, delimiter=',')
    savetxt(file_path_y, Y, delimiter=',')
    savetxt(file_path_X_train, X_train, delimiter=',')
    savetxt(file_path_y_train, y_train, delimiter=',')
    savetxt(file_path_X_test, X_test, delimiter=',')
    savetxt(file_path_y_test, y_test, delimiter=',')


def x_y_split(df):
    # print("Columns", df.columns)
    # Separating the target variable
    X = df.values[:, 0:(len(df.columns) - 1)]
    Y = df.values[:, -1]
    # print("X", X)
    # print("Y", Y)
    return X, Y


def x_y_train_test_split_and_store(df, test_size):
    X, Y = x_y_split(df)
    file_path_X = '../../data/processed/X.csv'
    file_path_y = '../../data/processed/y.csv'
    savetxt(file_path_X, X, delimiter=',')
    savetxt(file_path_y, Y, delimiter=',')

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=100)
    store_sets(X_train, X_test, y_train, y_test, df.columns)
    return X, Y, X_train, X_test, y_train, y_test


def vehicle_x_y_train_test_split_and_store(df, test_size):
    X, Y = x_y_split(df)
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=100)
    store_vehicle_sets(X, Y, X_train, X_test, y_train, y_test)
    return X, Y, X_train, X_test, y_train, y_test


def store_validation_sets(X_validation, y_validation):
    file_path_X_validation = '../../data/processed/X_validation.csv'
    file_path_y_validation = '../../data/processed/y_validation.csv'
    savetxt(file_path_X_validation, X_validation, delimiter=',')
    savetxt(file_path_y_validation, y_validation, delimiter=',')


def x_y_validation_split_and_store(df):
    X, Y = x_y_split(df)
    store_validation_sets(X, Y)


def load_data():
    dict = {}
    list_files = ['training_set', 'validation_set',
                  'validation_set_decision_tree_pruning',
                  'test_set_naive_bayes']
    for file in list_files:
        file_path = f'../../data/raw/{file}.csv'
        df = pd.read_csv(file_path)
        del df['Instance']
        # check_dataset(df)
        dict[file] = df
    df_training_set = dict['training_set']
    df_validation_set = dict['validation_set']
    df_validation_set_decision_tree_pruning = dict['validation_set_decision_tree_pruning']
    df_test_set_naive_bayes = dict['test_set_naive_bayes']

    return df_training_set, df_validation_set, df_validation_set_decision_tree_pruning, df_test_set_naive_bayes


def store_x_test_set(df, file_name):
    # delete target column with NaNs
    del df[df.columns[-1]]
    file_path_test_set = f'../../data/processed/{file_name}.csv'
    savetxt(file_path_test_set, df, delimiter=',')


def encode_sets(df_training_set, df_validation_set, df_validation_set_decision_tree_pruning, df_test_set_naive_bayes):
    df = df_training_set.append(df_validation_set)
    df = df.append(df_validation_set_decision_tree_pruning)
    df = df.append(df_test_set_naive_bayes)
    df_features = pd.get_dummies(df[[df.columns[0],
                                     df.columns[1]]],
                                 prefix={'Education Level': 'education_level',
                                         'Career': 'career'},
                                 drop_first=True)
    df_features[df.columns[2]] = ordinal_encode(df, ['Less than 3', '3 to 10', 'More than 10'], 2)
    df_features[df.columns[-1]] = ordinal_encode(df, ['Low', 'High'], 3)
    df_features.reset_index(drop=True)
    df_training_set = df_features.iloc[0:len(df_training_set)]
    df_validation_set = df_features.iloc[len(df_training_set):(len(df_training_set) +
                                                               len(df_validation_set))]

    df_validation_set_decision_tree_pruning = df_features.iloc[(len(df_training_set) +
                                                                len(df_validation_set)):(len(df_training_set) +
                                                                                         len(df_validation_set) +
                                                                                         len(df_validation_set_decision_tree_pruning))]
    df_test_set_naive_bayes = df_features.iloc[(len(df_training_set) +
                                                len(df_validation_set) +
                                                len(df_validation_set_decision_tree_pruning)):]

    return df_training_set, df_validation_set, df_validation_set_decision_tree_pruning, df_test_set_naive_bayes


def split_and_store_sets(df_training_set, df_validation_set,
                         df_test_set_decision_tree_pruning,  # same as validation set.
                         df_test_set_naive_bayes, test_size):
    x_y_train_test_split_and_store(df_training_set, test_size)  # first tree
    x_y_validation_split_and_store(df_validation_set)  # pruning tree
    store_x_test_set(df_test_set_naive_bayes, 'x_test_set_naive_bayes')


def load_salary_sets():
    df_training_set, df_validation_set, df_validation_set_decision_tree_pruning, df_test_set_naive_bayes = load_data()
    pie_chart(df_training_set[df_training_set.columns[-1]])

    print(df_training_set)
    print(df_validation_set)
    print(df_validation_set_decision_tree_pruning)
    print(df_test_set_naive_bayes)

    df_training_set, df_validation_set, df_validation_set_decision_tree_pruning, \
    df_test_set_naive_bayes = encode_sets(
        df_training_set,
        df_validation_set,
        df_validation_set_decision_tree_pruning,
        df_test_set_naive_bayes)

    print(df_training_set)
    print(df_validation_set)
    print(df_validation_set_decision_tree_pruning)
    print(df_test_set_naive_bayes)

    TEST_SIZE = 0.3
    split_and_store_sets(df_training_set,
                         df_validation_set,
                         df_validation_set_decision_tree_pruning,
                         df_test_set_naive_bayes, TEST_SIZE)


def load_vehicle_set():
    file_path = f'../../data/raw/veh-prime.arff'
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df


def encode_class_and_store(df):
    replace_values = {b'noncar': 0, b'car': 1}
    df[df.columns[-1]] = df[df.columns[-1]].replace(replace_values)
    file_path_vehicle_df = '../../data/interim/vehicle/vehicle_df.pickle'
    open_file = open(file_path_vehicle_df, "wb")
    pickle.dump(df, open_file)
    open_file.close()
    return df


def calculate_correlation(df):
    feature_list = df.columns[:-1]
    target_variable = df.columns[-1]
    dict = {}
    for feature in feature_list:
        # [0] because it returns first r and p-value.
        dict[feature] = scipy.stats.pearsonr(df[feature], df[target_variable])[0]
    return dict


def list_features_from_highest_correlation(df):
    dict_correlation = calculate_correlation(df)
    dict_correlation = {k: abs(v) for k, v in
                        sorted(dict_correlation.items(), key=lambda item: abs(item[1]), reverse=True)}
    print("The correlation of the features with the target variable (car or not car), from highest absolute "
          "correlation to "
          "lowest is: \n", dict_correlation)

    features_correlation = pd.DataFrame(dict_correlation.items(), columns=['Features', 'Pearson Correlation'])
    file_path_dict_correlation = '../../data/interim/vehicle/features_correlation.pickle'
    open_file = open(file_path_dict_correlation, "wb")
    pickle.dump(features_correlation, open_file)
    open_file.close()


def load_encode_and_calculate_correlation_vehicle_set():
    df = load_vehicle_set()
    df = encode_class_and_store(df)
    list_features_from_highest_correlation(df)
    return df


def process_vehicle_data():
    df_training_set = load_encode_and_calculate_correlation_vehicle_set()
    test_size = 0.3
    vehicle_x_y_train_test_split_and_store(df_training_set, test_size)


def main():
    load_salary_sets()
    process_vehicle_data()


if __name__ == '__main__':
    main()
