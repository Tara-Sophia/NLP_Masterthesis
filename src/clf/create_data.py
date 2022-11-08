# -*- coding: utf-8 -*-
"""
Description:
   Loading data and transforming data 
   to the format that can be used by the classifier.
"""

import ast
import os

import pandas as pd
from constants import RAW_DATA_DIR, X_MASKED
from sklearn.model_selection import train_test_split


# transform input data for model
def replace_tab(x):
    return [i.replace(" ", "_") for i in x]


def transform_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Transform column to list

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with labels and NLP features
    column_name : str
        column name

    Returns
    -------
    pd.DataFrame
        dataframe with transformed column
    """
    df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x))
    df[column_name] = df[column_name].apply(lambda x: replace_tab(x))
    df[column_name] = df[column_name].apply(lambda x: " ".join(x))
    return df


def main():
    # Load data
    df = pd.read_csv(RAW_DATA_DIR)
    df = transform_column(df, X_MASKED)
    print(df.shape)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.transcription_f_semisupervised.to_list(),
        df.medical_specialty,
        test_size=0.2,
        random_state=42,
    )

    # Save data as csv
    train_df = pd.DataFrame({"keywords": X_train, "medical_specialty": y_train})
    test_df = pd.DataFrame({"keywords": X_test, "medical_specialty": y_test})
    train_df.to_csv(
        os.path.join("data", "processed", "clf", "train.csv"),
        index=False,
    )
    test_df.to_csv(
        os.path.join("data", "processed", "clf", "test.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
