# Start the experiment by creating 3 additional training files from the train-1000-100.csv by taking the first 50,
# 100, and 150 instances respectively. Call them: train-50(1000)-100.csv, train-100(1000)- 100.csv, train-150(
# 1000)-100.csv. The corresponding test file for these dataset would be test-1000- 100.csv and no modification is
# needed.
import pandas as pd


def check_dataset(df):
    """

    Args:
        df: dataframe to be analyzed.

    Returns: Quick analysis of the dataset. Duplicated, missing values, statistics, etc.

    """
    print(df.columns)
    print(df.shape, '\n')
    # print(df.head(), '\n')
    print('Describe:\n', df.describe(), '\n')
    print('Info:\n', df.info(), '\n')
    # print('Counts:\n', df.value_counts(), '\n')
    print("Duplicate Rows :\n", df[df.duplicated()], '\n')
    print('\n')
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    print("Missing Values:", missing_value_df.sort_values('percent_missing', inplace=True), '\n')


def create_training_test_files():
    """

    Returns:
        .csv files: stores training and test files to the data folder.

    """
    file_path = '../../data/raw/train-1000-100.csv'
    file_path_test = '../../data/raw/test-1000-100.csv'
    df_train = pd.read_csv(file_path)
    file_sizes = [50, 100, 150]

    for size in file_sizes:
        file_path_train = f'../../data/raw/train-{size}(1000)-100.csv'
        df_train.iloc[:size].to_csv(file_path_train, index=False)
    df_test = pd.read_csv(file_path_test)

    file_path_test = '../../data/raw/test-1000-100.csv'
    df_test.iloc[:size].to_csv(file_path_test, index=False)


def check_training_data():
    """

    Returns: Quick analysis of the dataset. Duplicated, missing values, statistics, etc.

    """
    file_sizes = [50, 100, 150]
    folder = '/raw/'
    for size in file_sizes:
        file = f'train-{size}(1000)-100.csv'
        file_path_train = '../../data' + folder + file
        df_train = pd.read_csv(file_path_train)
        print('Check for:', file)
        check_dataset(df_train)

    file_sizes = [10, 100]
    for size in file_sizes:
        file = f'train-100-{size}.csv'
        file_path_train = '../../data' + folder + file
        df_train = pd.read_csv(file_path_train)
        print('Check for:', file)
        check_dataset(df_train)

    file = 'train-1000-100.csv'
    file_path_train = '../../data' + folder + file
    df_train = pd.read_csv(file_path_train)
    print('Check for:', file)
    check_dataset(df_train)


def main():
    """

    Returns:
        .csv files: with training and testing data.


    """
    create_training_test_files()
    check_training_data()


if __name__ == '__main__':
    main()
