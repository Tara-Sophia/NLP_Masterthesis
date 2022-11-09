# -*- coding: utf-8 -*-
# cleaning the dataframe
import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from constants import (
    MTSAMPLES_PROCESSED_PATH_DIR,
    MTSAMPLES_RAW_PATH_DIR,
)
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")

nltk.download("wordnet")
nltk.download("omw-1.4")


def cleaning_input(sentence: str) -> str:
    """
    This function cleans the input sentence by removing stopwords, numbers and punctuation, and lemmatizing the words.

    Parameters
    ----------
    sentence : str
        The sentence to be cleaned.

    Returns
    -------
    str
        The cleaned sentence.
    """
    # Basic cleaning
    sentence = sentence.strip()  ## remove whitespaces
    sentence = sentence.lower()  ## lowercase
    sentence = "".join(
        char for char in sentence if not char.isdigit()
    )  ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")  ## remove punctuation

    tokenized_sentence = word_tokenize(sentence)  ## tokenize
    stop_words = set(stopwords.words("english"))  ## define stopwords

    tokenized_sentence_cleaned = [  ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words
    ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos="v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = " ".join(word for word in lemmatized)

    return cleaned_sentence


def create_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops nan columns, applies the cleaning function and creates a column with the keywords as list

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
    # df["location"] = df.apply(
    #     lambda x: location_indices(x.transcription, x.keywords_list), axis=1
    return df


def top_11_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function keeps only the top 11 classes, based on count, of the dataframe.
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
    top_11 = df.value_counts("medical_specialty").index[:11]
    return df[df["medical_specialty"].isin(top_11)].copy()


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
    This function loads the dataframe, cleans it and saves it to a csv file.
    """

    df = pd.read_csv(os.path.join(MTSAMPLES_RAW_PATH_DIR, "mtsamples.csv"))
    df = create_df(df)
    df = top_11_classes(df)

    # save dataframe
    save_df(df, MTSAMPLES_PROCESSED_PATH_DIR)


if __name__ == "__main__":
    main()
