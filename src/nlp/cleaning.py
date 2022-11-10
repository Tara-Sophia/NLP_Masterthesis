# -*- coding: utf-8 -*-
# cleaning the dataframe
import os
import string

import nltk
import pandas as pd
from constants import (
    MTSAMPLES_PROCESSED_PATH_DIR,
    MTSAMPLES_RAW_PATH_DIR,
    most_common_words_filtered,
)
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import cleaning_input

nltk.download("stopwords")
nltk.download("punkt")

nltk.download("wordnet")
nltk.download("omw-1.4")


def create_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops nan columns, applies the cleaning function.
    Creates a column with the keywords as list.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be worked on.

    Returns
    -------
    pd.DataFrame
        The cleaned, without nan values dataframe.
    """
    df = df.dropna().copy()
    df["transcription"] = df["transcription"].apply(cleaning_input)

    df["keywords_list"] = df["keywords"].apply(lambda x: x.split(","))
    return df


def top_classes(df: pd.DataFrame) -> pd.DataFrame:
    # more than 100 samples
    """
    This function keeps only the top 11 classes,
    based on count of the dataframe.
    The reason is that the other classes are too few to be used for training.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be worked on.

    Returns
    -------
    pd.DataFrame
        The dataframe with only the top 11 classes.
    """
    # dataframe filter for classed with more than 100 samples
    df = df[
        df["medical_specialty"].isin(
            df["medical_specialty"]
            .value_counts()[df["medical_specialty"].value_counts() > 100]
            .index
        )
    ]
    # top_11 = df.value_counts("medical_specialty").index[:11]
    # return df[df["medical_specialty"].isin(top_11)].copy()
    return df


def create_dir(path: str) -> None:
    """
    This function creates a directory if it does not exist.

    Parameters
    ----------
    path : str
        The path of the directory to be created.
    """
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, path: str) -> None:
    """
    This function saves the dataframe to a csv file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be saved.
    path : str
        The path of the directory where the dataframe will be saved.
    """
    create_dir(path)
    df.to_csv(os.path.join(path, "mtsamples_cleaned.csv"), index=False)


def main() -> None:
    """
    Main function
    """

    df = pd.read_csv(os.path.join(MTSAMPLES_RAW_PATH_DIR, "mtsamples.csv"))
    df = create_df(df)
    df = top_classes(df)

    # Save dataframe
    save_df(df, MTSAMPLES_PROCESSED_PATH_DIR)


if __name__ == "__main__":
    main()
