import os
import shutil
import librosa
from sklearn.model_selection import train_test_split
import pandas as pd

REL_PATH_WAV = os.path.join(
    "data", "raw", "stt"
)  # These can change, depending on the data we are using
REL_PATH_RECORDINGS = os.path.join(
    "data", "raw", "stt"
)  # These can change, depending on the data we are using

CSV_FILE = os.path.join(
    REL_PATH_RECORDINGS, "overview-of-recordings.csv"
)
WAV_FILES_DIR = os.path.join(REL_PATH_WAV, "recordings")

MAX_DURATION_LENGTH = 4.5


def copy_files(files_list, folder):
    new_dir = os.path.join(REL_PATH_WAV, folder)
    if os.path.exists(new_dir) and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)

    for file in files_list:
        shutil.copyfile(
            os.path.join(WAV_FILES_DIR, file),
            os.path.join(new_dir, file),
        )


def get_wav_file_duration(file_path):
    return librosa.get_duration(
        filename=os.path.join(WAV_FILES_DIR, file_path)
    )


def clean_df(df, folder):

    df["path"] = df["file_name"].apply(
        lambda x: os.path.join(REL_PATH_WAV, folder, x)
    )
    df["audio"] = df["path"]
    df = df.rename(columns={"phrase": "sentence"})

    copy_files(df["file_name"].tolist(), folder)

    df = df.loc[:, ["audio", "sentence", "path"]]

    return df


def create_own_dataset(file_path):

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


df_train, df_val, df_test = create_own_dataset(CSV_FILE)
df_train.to_csv(
    os.path.join(REL_PATH_RECORDINGS, "train.csv"), index=False
)
df_val.to_csv(
    os.path.join(REL_PATH_RECORDINGS, "val.csv"), index=False
)
df_test.to_csv(
    os.path.join(REL_PATH_RECORDINGS, "test.csv"), index=False
)