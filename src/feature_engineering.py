import itertools
import pandas as pd
import xgboost as xgb
import copy

from sklearn import metrics
from sklearn import preprocessing

def feature_engineering_cat(df, cat_cols):
    """
    This Function is used for feature engineering
    :params df: the pandas dataframe with train/test data
    :params cat_tools: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values
    # in the list
    # for example
    # list(itertools.combinations([1, 2, 3], 2)) will return
    # [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def mean_target_encoding(data):
    """
        Applies mean target encoding to the given data.

        Parameters:
            data (pandas.DataFrame): The input data to apply mean target encoding on.

        Returns:
            pandas.DataFrame: The encoded data with additional columns for each feature representing the encoded values.

        Description:
            This function applies mean target encoding to the given data. Mean target encoding replaces the categorical
            values of a feature with the average target value for that category. It is commonly used in machine learning
            for encoding categorical features.

            The function takes a pandas DataFrame as input, which represents the data to be encoded. It first creates a
            deep copy of the data to avoid modifying the original data. It then defines a list of numerical columns to
            exclude from encoding.

            Next, the function maps the target values to 0s and 1s using a mapping dictionary. It replaces the '
    """

    df = copy.deepcopy(data)

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # map targets to 0s and 1s
    target_mapping = {
        "<=50" : 0,
        ">50K" : 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)

    features = [
        f for f in df.columns if f not in ("kfold", "income")
        and f not in num_cols
    ]

    # fill NaN values with NONE
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])

            df.loc[:, col] = lbl.transform(df.loc[col])

    encoded_dfs = []

    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop = True)
        df_valid = df[df.kfold == fold].reset_index(drop = True)

        for column in features:
            mapping_dict = dict(
                df_train.groupby(column)["income"].mean()
            )
            df_valid.loc[
                :, column + "_enc"
            ] = df_valid[column].map(mapping_dict)
        encoded_dfs.append(df_valid)
    encoded_dfs = pd.concat(encoded_dfs, axis = 0)

    return encoded_dfs