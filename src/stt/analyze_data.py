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
import numpy as np
import pandas as pd
from constants import (
    RAW_RECORDINGS_DIR,
    RECORDINGS_FILE,
    STT_REPORT,
    SRC_DIR,
)
from pandas_profiling import ProfileReport

sys.path.insert(0, SRC_DIR)
from decorators import log_function_name

# ! Todo
# # Load files
# audio_segment = AudioSegment.from_file(
#     "Downloads/Warm-Memories-Emotional-Inspiring-Piano.wav"
# )
# # Print attributes
# print(f"Channels: {audio_segment.channels}")
# print(f"Sample width: {audio_segment.sample_width}")
# print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
# print(f"Frame width: {audio_segment.frame_width}")
# print(f"Length (ms): {len(audio_segment)}")
# print(f"Frame count: {audio_segment.frame_count()}")
# print(f"Intensity: {audio_segment.dBFS}")
# print(f"Get duration: {audio_segment.duration_seconds}")
# # librsa.feature.mfcc


def get_librosa_features(row: pd.Series) -> tuple[int, float]:
    """
    Get the duration of a wav file

    Parameters
    ----------
    row : pd.Series
        Row of the dataframe

    Returns
    -------
    tuple[int, float]
        Sample rate and duration of the wav file
    """
    file_path = row.file_location
    sr = librosa.get_samplerate(file_path)
    duration = librosa.get_duration(filename=file_path)
    return sr, np.round(duration, 3)


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
    df["file_location"] = df["file_name"].apply(
        lambda x: os.path.join(RAW_RECORDINGS_DIR, x)
    )
    df[["sr", "duration"]] = df.apply(
        lambda x: get_librosa_features(x),
        result_type="expand",
        axis=1,
    )
    return df


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
def main():
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
    df = load_dataframe(RECORDINGS_FILE, important_columns)
    df = transform_dataframe(df)
    create_dir(STT_REPORT)
    report = make_report(df)
    save_report(report, STT_REPORT)


if __name__ == "__main__":
    main()
