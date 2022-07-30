# -*- coding: utf-8 -*-

import pandas as pd


def check_dataset(df):
    print(df.shape, '\n')
    print(df.head(), '\n')
    print(df.describe(), '\n')
    print(df.info(), '\n')
    print(df.value_counts(), '\n')
    print("Duplicate Rows :", df[df.duplicated()])
    print('\n')
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    print("Missing Values:", missing_value_df.sort_values('percent_missing', inplace=True))
    print('\n')


def main():
    """

    Returns:
        - Return Data transformed from the raw data, to proceed with the analysis later on.
        - Store transformed data it into the /interim folder.

    """
    file_path_train = '../../data/raw/train.csv'
    file_path_test = '../../data/raw/test.csv'

    df_train = pd.read_csv(file_path_train)
    df_test = pd.read_csv(file_path_test)

    # data wrangling on training and test data.
    columns = ['id', 'ExampleID']
    df_train.drop(columns, axis=1, inplace=True)
    df_test.drop(columns, axis=1, inplace=True)

    check_dataset(df_train)
    check_dataset(df_test)

    # duplicated rows detected and erased
    df_train.drop_duplicates(inplace=True)

    # storing pre-processed data in the interim folder
    df_train.to_pickle('../../data/interim/train_df.pkl')
    df_test.to_pickle('../../data/interim/test_df.pkl')


if __name__ == '__main__':
    main()
