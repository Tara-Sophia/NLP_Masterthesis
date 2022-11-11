# -*- coding: utf-8 -*-

"""
Description:
    Creating a PDF report about the features of the dataset

Usage:
    $ python src/data/analyze_data.py
"""
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import (
    RAW_DATA_DIR,
    RECORDINGS_FILE,
    SRC_DIR,
    STT_REPORT,
)
from pandas_profiling import ProfileReport

sys.path.insert(0, SRC_DIR)
from decorators import log_function_name  # noqa: E402


def get_librosa_features(
    row: pd.Series,
) -> tuple[np.ndarray, int, float]:
    """
    Get the duration of a wav file

    Parameters
    ----------
    row : pd.Series
        Row of the dataframe

    Returns
    -------
    tuple[np.ndarray, int, float]
        Audio array list, sample rate and duration of the wav file
    """
    file_path = row.path
    audio, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(filename=file_path)
    return audio, sr, np.round(duration, 3)


@log_function_name
def load_dataframe(filename: str, columns: list[str]) -> pd.DataFrame:
    """
    Load the dataframe from the csv file

    Parameters
    ----------
    filename : str
        Name of the csv file
    columns : list[str]
        List of columns to load

    Returns
    -------
    pd.DataFrame
        Dataframe with the loaded data
    """
    df = pd.read_csv(filename, usecols=columns)
    return df


@log_function_name
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to transform

    Returns
    -------
    pd.DataFrame
        Transformed dataframe
    """
    df[["audio", "sr", "duration"]] = df.apply(
        lambda x: get_librosa_features(x),
        result_type="expand",
        axis=1,
    )
    return df


@log_function_name
def load_train_val_test_data(folder_path: str) -> pd.DataFrame:
    """
    Load the train, validation and test data

    Parameters
    ----------
    folder_path : str
        Path to the folder with the train, validation and test data

    Returns
    -------
    pd.DataFrame
        Dataframe with the concatenated train, validation and test data
    """
    train_df = pd.read_csv(os.path.join(folder_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(folder_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(folder_path, "test.csv"))

    train_df["split_type"] = "train"
    val_df["split_type"] = "val"
    test_df["split_type"] = "test"

    df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(
        drop=True
    )
    df["file_name"] = df["path"].apply(lambda x: os.path.split(x)[1])
    df = transform_dataframe(df)
    return df


@log_function_name
def evaluate_audio_sample(row: pd.Series) -> None:
    """
    Evaluate the audio sample

    Parameters
    ----------
    row : pd.Series
        Row of the dataframe
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    librosa.display.waveshow(row.audio, sr=row.sr, ax=ax[0])
    ax[0].set(title="Waveplot")
    ax[1].specgram(row.audio, Fs=row.sr)
    ax[0].set(ylabel="Amplitude", title="Waveplot")
    ax[1].set(
        xlabel="Time (seconds)",
        ylabel="Frequency (HZ)",
        title="Spectogram",
    )
    fig.suptitle(f"Text: {row.phrase}")
    plt.savefig(os.path.join(STT_REPORT, "audio_sample.png"))


@log_function_name
def create_dir(file_path: str) -> None:
    """
    Create the directory if it does not exist

    Parameters
    ----------
    file_path : str
        Path to the directory
    """
    os.makedirs(file_path, exist_ok=True)


@log_function_name
def create_charts(
    df: pd.DataFrame,
    col_use: str,
    title: str,
    xlabel: str,
    ylabel: str,
    hist: bool,
    file_name: str,
) -> None:
    """
    Create charts to visualize the difference between the train, val and test

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to use
    col_use : str
        Column to visualize
    title : str
        Title of the chart
    xlabel : str
        X label of the chart
    ylabel : str
        Y label of the chart
    hist : bool
        Flag to indicate if the chart is a histogram
    file_name : str
        Name of the file to save
    """
    train = df.loc[df["split_type"] == "train", col_use]
    val = df.loc[df["split_type"] == "val", col_use]
    test = df.loc[df["split_type"] == "test", col_use]

    plt.figure(figsize=(12, 8))

    if hist:
        # Add three histograms to one plot
        plt.hist(train, alpha=0.3, label="train", color="blue")
        plt.hist(val, alpha=0.3, label="val", color="orange")
        plt.hist(test, alpha=0.3, label="test", color="green")

        # Add legend
        plt.legend(title="Split Type")
    else:
        plt.bar(
            df["split_type"].unique(),
            height=df["split_type"].value_counts(),
            alpha=0.3,
            color=[
                "blue",
                "orange",
                "green",
            ],
        )

    # Add plot title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(os.path.join(STT_REPORT, file_name))


@log_function_name
def make_report(df: pd.DataFrame) -> ProfileReport:
    """
    Create a report of the dataframe with pandas profiling

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to create the report for

    Returns
    -------
    ProfileReport
        Report of the dataframe
    """
    report = ProfileReport(df, title="STT Report", explorative=True)
    return report


@log_function_name
def save_report(report: ProfileReport, file_path: str) -> None:
    """
    Save the report to the file path as an html file

    Parameters
    ----------
    report : ProfileReport
        Report to save
    file_path : str
        Path to save the report to
    """
    report.to_file(os.path.join(file_path, "report.html"))


@log_function_name
def main() -> None:
    """
    Main function
    """
    important_columns = [
        "audio_clipping",
        "overall_quality_of_the_audio",
        "quiet_speaker",
        "speaker_id",
        "file_name",
        "phrase",
        "prompt",
        "writer_id",
    ]
    # Load main dataframe
    df_full = load_dataframe(RECORDINGS_FILE, important_columns)

    # Train, val, test evaulation
    df_train_val_test = load_train_val_test_data(RAW_DATA_DIR)

    # Merge dataframes
    df = pd.merge(
        df_full, df_train_val_test, on="file_name", how="inner"
    )
    print(df.shape)

    # Create directory for the report and figures
    create_dir(STT_REPORT)

    # Create charts
    create_charts(
        df,
        "sr",
        "Original Sample Rate by Split",
        "Sample Rate",
        "Frequency",
        True,
        "sample_rate.png",
    )
    create_charts(
        df,
        "duration",
        "Duration by Split",
        "Duration in seconds",
        "Frequency",
        True,
        "duration.png",
    )
    create_charts(
        df,
        "split_type",
        "Split Type",
        "Split Type",
        "Frequency",
        False,
        "split_type_numbers.png",
    )

    # Evaluate sample audio file
    # The 6th example is a good one to show
    df_sample = df.iloc[6, :]
    evaluate_audio_sample(df_sample)

    # Create report
    report = make_report(df)
    save_report(report, STT_REPORT)


if __name__ == "__main__":
    main()
