import os
import sys
import pandas as pd
from constants import (
    ICD9_SPECIALTY_DICT,
    NOTEEVENTS_COLS,
    MIMIC_DIAGNOSES_CSV,
    ICD9_CSV,
    MIMIC_NOTEEVENTS_CSV,
    DIAGNOSES_NOTEEVENTS_CSV,
)

sys.path.insert(0, "src")
from decorators import log_function_name


@log_function_name
def clean_mimic_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ICD9_CODE"], axis=0).copy()
    df = df.sort_values("SEQ_NUM", ascending=True)
    df = df.drop_duplicates(
        subset=["SUBJECT_ID", "HADM_ID"], keep="first"
    )
    df["ICD9_CODE"] = df["ICD9_CODE"].str[:3]
    return df


@log_function_name
def clean_icd9(df: pd.DataFrame) -> pd.DataFrame:
    df["category_codes"] = df["category_codes"].str.rjust(3, "0")
    df.loc[
        df["category_codes"].str.startswith("E"), "category_codes"
    ] = df["category_codes"].str[:3]
    df = df.drop_duplicates(
        subset=["category_codes", "chapter_codes_des"], keep="first"
    ).copy()
    return df


@log_function_name
def clean_mimic_noteevents(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["ISERROR"].isna())
        & (df["CATEGORY"] == "Discharge summary")
    ].copy()
    df = df.drop_duplicates(
        subset=["SUBJECT_ID", "HADM_ID"], keep="first"
    ).copy()
    return df


@log_function_name
def merge_diagnoses_icd9(
    df_mimic_diagnoses: pd.DataFrame, df_icd9: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(
        df_mimic_diagnoses,
        df_icd9,
        left_on="ICD9_CODE",
        right_on="category_codes",
        how="left",
    )
    df["specialty"] = df["chapter_codes_des"].map(ICD9_SPECIALTY_DICT)
    df = df.drop(
        columns=[
            "category_codes",
            "category_codes_des",
            "block_codes",
            "block_codes_des",
            "chapter_codes",
            "chapter_codes_des",
        ]
    ).copy()
    return df


@log_function_name
def merge_diagnoses_noteevents(
    df_diagnoses_icd9: pd.DataFrame, df_mimic_noteevents: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(
        df_diagnoses_icd9,
        df_mimic_noteevents,
        on=["SUBJECT_ID", "HADM_ID"],
        how="left",
    )
    df = df[["TEXT", "specialty"]].copy()
    return df


@log_function_name
def save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


@log_function_name
def main():
    """
    Main function
    """
    # LOAD
    df_mimic_diagnoses = pd.read_csv(MIMIC_DIAGNOSES_CSV)
    df_icd9 = pd.read_csv(ICD9_CSV)
    df_mimic_noteevents = pd.read_csv(
        MIMIC_NOTEEVENTS_CSV, usecols=NOTEEVENTS_COLS
    )

    # CLEAN
    df_mimic_diagnoses = clean_mimic_diagnoses(df_mimic_diagnoses)
    df_icd9 = clean_icd9(df_icd9)
    df_mimic_noteevents = clean_mimic_noteevents(df_mimic_noteevents)

    # MERGE
    df_diagnoses_icd9 = merge_diagnoses_icd9(
        df_mimic_diagnoses, df_icd9
    )
    df_diagnoses_noteevents = merge_diagnoses_noteevents(
        df_diagnoses_icd9, df_mimic_noteevents
    )

    # SAVE
    save_df(df_diagnoses_noteevents, DIAGNOSES_NOTEEVENTS_CSV)


if __name__ == "__main__":
    main()
