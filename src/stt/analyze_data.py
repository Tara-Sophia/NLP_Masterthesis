# -*- coding: utf-8 -*-
import os

import librosa
import numpy as np
import pandas as pd
from constants import CSV_FILE, REL_PATH_RECORDINGS, STT_REPORT
from pandas_profiling import ProfileReport

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


def get_librosa_features(row):
    file_path = row.file_location
    sr = librosa.get_samplerate(file_path)
    duration = librosa.get_duration(filename=file_path)
    return sr, np.round(duration, 3)


def load_dataframe(filename, columns):
    df = pd.read_csv(filename, usecols=columns)
    return df


def transform_dataframe(df):
    df["file_location"] = df["file_name"].apply(
        lambda x: os.path.join(REL_PATH_RECORDINGS, "recordings", x)
    )
    df[["sr", "duration"]] = df.apply(
        lambda x: get_librosa_features(x),
        result_type="expand",
        axis=1,
    )
    return df


def create_dir(file_path):
    os.makedirs(file_path, exist_ok=True)


def make_report(df):
    report = ProfileReport(df, title="STT Report", explorative=True)
    return report


def save_report(report, file_path):
    report.to_file(os.path.join(file_path, "report.html"))


def main():
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
    df = load_dataframe(CSV_FILE, important_columns)
    df = transform_dataframe(df)
    create_dir(STT_REPORT)
    report = make_report(df)
    save_report(report, STT_REPORT)


if __name__ == "__main__":
    main()
