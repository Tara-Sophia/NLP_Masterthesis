# -*- coding: utf-8 -*-

"""
Description:
    Creation of the speech data by splitting the recordings
    into train, validation and test sets.
    Filtering out recordings that are too long.

Usage:
    $ python src/data/create_speech_data.py
"""
import os
import shutil

import librosa
import pandas as pd
from constants import (
    MAX_DURATION_LENGTH,
    RAW_DATA_DIR,
    RAW_RECORDINGS_DIR,
    RECORDINGS_FILE,
)
from decorators import log_function_name
from sklearn.model_selection import train_test_split


def get_wav_file_duration(file_path: str) -> float:
    """
    Get the duration of a wav file

    Parameters
    ----------
    file_path : str
        Path to the wav file

    Returns
    -------
    float
        Duration of the wav file in seconds
    """
    return librosa.get_duration(
        filename=os.path.join(RAW_RECORDINGS_DIR, file_path)
    )


@log_function_name
def copy_files(files_list: list[str], folder: str) -> None:
    """
    Copy files from the raw data folder to the new folder

    Parameters
    ----------
    files_list : list[str]
        List of files to copy
    folder : str
        Name of the folder to copy the files to
    """
    new_dir = os.path.join(RAW_DATA_DIR, folder)
    if os.path.exists(new_dir) and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)

    for file in files_list:
        shutil.copyfile(
            os.path.join(RAW_RECORDINGS_DIR, file),
            os.path.join(new_dir, file),
        )


@log_function_name
def clean_df(df: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Clean the dataframe by adjusting, adding and removing the required columns.
    Adding the files to the new respective folder.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to clean
    folder : str
        Name of the folder to copy the files to

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """

    df["path"] = df["file_name"].apply(
        lambda x: os.path.join(RAW_DATA_DIR, folder, x)
    )
    df["audio"] = df["path"]
    df = df.rename(columns={"phrase": "sentence"})

    copy_files(df["file_name"].tolist(), folder)

    df = df.loc[:, ["audio", "sentence", "path"]]

    return df


@log_function_name
def create_own_dataset(
    file_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Adjusting the dataset to the required format

    Parameters
    ----------
    file_path : str,
        Path to the csv file

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Tuple of train, validation and test dataframes
    """

    df = pd.read_csv(file_path)

    df["duration"] = df["file_name"].apply(get_wav_file_duration)
    df = df[df["duration"] < MAX_DURATION_LENGTH].copy()

    df_train, df_test = train_test_split(
        df, test_size=0.25, random_state=42
    )
    df_train, df_val = train_test_split(
        df_train, test_size=0.3, random_state=42
    )

    df_train = clean_df(df_train, "train")
    df_val = clean_df(df_val, "val")
    df_test = clean_df(df_test, "test")

    return df_train, df_val, df_test


@log_function_name
def main() -> None:
    """
    Main function
    """
    df_train, df_val, df_test = create_own_dataset(RECORDINGS_FILE)
    df_train.to_csv(
        os.path.join(RAW_DATA_DIR, "train.csv"), index=False
    )
    df_val.to_csv(os.path.join(RAW_DATA_DIR, "val.csv"), index=False)
    df_test.to_csv(
        os.path.join(RAW_DATA_DIR, "test.csv"), index=False
    )


if __name__ == "__main__":
    main()
