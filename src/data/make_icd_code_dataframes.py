# -*- coding: utf-8 -*-

"""
Description:
    Scraping symptoms from Wikipedia and NHS and adding to ICD codes dataframe.
    The data will be saved in data/interim folder.

Usage:
    $ python src/data/make_icd_codes_dataframes.py -a
Possible arguments:
    * -i9 or --icd9: ICD-9 codes
    * -i10 or --icd10: ICD-10 codes
    * -a or -all: ICD-9 and ICD-10 codes
"""

import os
from pathlib import Path
from typing import Callable

import click
import pandas as pd
import simple_icd_10_cm as cm

MAX_TABS = 3


def save_df(df: pd.DataFrame, file_path: str) -> None:
    """
    Saving a dataframe as a csv file

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which should be saved
    file_path : str
        Path to storage location of the csv file
    """
    df = df[
        [
            "category_codes",
            "category_codes_des",
            "block_codes",
            "block_codes_des",
            "chapter_codes",
            "chapter_codes_des",
        ]
    ]
    df.to_csv(file_path, index=False)


# ICD 10


def check_valid(ser: pd.Series, func: Callable) -> pd.Series:
    """
    Check if the values in a series are valid based on their ICD code

    Parameters
    ----------
    ser : pd.Series
        Series of ICD codes
    func : Callable
        Check if ICD codes are valid

    Returns
    -------
    pd.Series
        Returns the series with the invalid values replaced with their furthers ancestor
    """
    for index, value in ser.items():
        if not func(value):
            ser[index] = cm.get_ancestors(value)[-1]
    return ser


def get_parent_vals(
    df: pd.DataFrame, child: str, parent: str, func: Callable
) -> pd.DataFrame:
    """
    Get parent ICD codes from child ICD codes

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with ICD codes
    child : str
        Child ICD code
    parent : str
        Name for parent ICD code
    func : Callable
        Function to check if ICD codes are valid

    Returns
    -------
    pd.DataFrame
        Dataframe with parent ICD codes
    """
    df[parent] = df[child].apply(cm.get_parent)
    df[parent] = check_valid(df[parent], func)
    df[f"{parent}_des"] = df[parent].apply(cm.get_description)
    return df


# ICD 9


def expand_codes_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Split ICD codes into codes and description

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with ICD codes and descriptions in one column
    col : str
        Column which should be split

    Returns
    -------
    pd.DataFrame
        Dataframe with ICD codes and descriptions in separate columns
    """
    df[[col, f"{col}_des"]] = df[col].str.split(" ", n=1, expand=True)
    return df


def create_dir(folder_path: str) -> None:
    """
    Create a directory if it does not exist

    Parameters
    ----------
    folder_path : str
        Path to the directory
    """
    p = Path(folder_path)
    p.mkdir(exist_ok=True)


def make_df_from_txt(file_path: str, csv_folder: str) -> None:
    """
    Read in txt file and save them as cleaned csv file

    Parameters
    ----------
    file_path : str
        Path to the txt file
    csv_folder : str
        Path to the folder where the csv files should be saved
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chapter_codes", "block_codes", "category_codes"],
    )

    # Check if chapter codes are unique
    if df.chapter_codes.nunique() == 1:

        # Check if category codes are given
        if df.category_codes.isnull().all():
            df.category_codes = df.block_codes

        # Fill chapter and block codes with previous values
        df[["chapter_codes", "block_codes"]] = df[
            ["chapter_codes", "block_codes"]
        ].ffill()
        df = df.dropna().reset_index(drop=True)

        # Split chapter codes into codes and description
        expand_codes_col(df, "chapter_codes")
        expand_codes_col(df, "block_codes")
        expand_codes_col(df, "category_codes")

        # Save dataframe
        save_df(
            df,
            os.path.join(csv_folder, Path(file_path).stem + ".csv"),
        )

    else:
        print("File has multiple chapter codes", file_path)


def clean_txt_files(file_path: str, lines: list[str]) -> None:
    """
    Clean txt files by appending tabs to the lines if necessary

    Parameters
    ----------
    file_path : str
        Path to the txt file
    lines : list[str]
        List of lines in the txt file
    """
    with open(file_path, "w") as f:
        for line in lines:
            tab_diff = MAX_TABS - line.count("\t")
            if tab_diff > 0:
                line = line.strip("\n") + "\t" * tab_diff + "\n"
            f.write(line)


def concat_and_save_df_from_csv(
    source_folder_path: str, target_file_path: str
) -> None:
    """
    Concatenate csv files and save them as a single csv file

    Parameters
    ----------
    source_folder_path : str
        Path to the folder with the csv files
    target_file_path : str
        Path to the csv file where the concatenated csv files should be saved
    """
    df = pd.DataFrame()
    for file in os.listdir(source_folder_path):
        df_part = pd.read_csv(os.path.join(source_folder_path, file))
        df = pd.concat([df, df_part], ignore_index=True)

    save_df(df, target_file_path)


def make_icd_9_csv(
    source_folder_path: str, target_folder_path: str
) -> None:
    """
    Create ICD 9 csv file

    Parameters
    ----------
    source_folder_path : str
        Path to the folder with the txt files
    target_folder_path : str
        Path to the folder where the ICD 9 csv file should be saved
    """
    # Creating icd9 folder
    icd_folder = os.path.join(target_folder_path, "icd9")

    # Creating directory for cleaned text files and csv files
    txt_folder = os.path.join(icd_folder, "txt")
    csv_folder = os.path.join(icd_folder, "csv")

    # Make sure directory exists
    create_dir(icd_folder)
    create_dir(txt_folder)
    create_dir(csv_folder)

    for file in os.listdir(source_folder_path):

        # Read all files in icd9 folder
        with open(f"{source_folder_path}/{file}", "r") as f:
            lines = f.readlines()

        # Save clenaed text files by adding efficient number of tabs
        clean_txt_files(os.path.join(txt_folder, file), lines)

        # Make dataframe from cleaned text files
        make_df_from_txt(os.path.join(txt_folder, file), csv_folder)

        # Concatenate all csv files into one dataframe and save file
        concat_and_save_df_from_csv(
            csv_folder,
            os.path.join(
                target_folder_path, "icd9_codes_and_des.csv"
            ),
        )


def make_icd_10_csv(target_folder_path: str) -> None:
    """
    Create ICD 10 csv file

    Parameters
    ----------
    target_folder_path : str
        Path to the folder where the ICD 10 csv file should be saved
    """
    # Get all ICD-10-CM codes
    all_codes = cm.get_all_codes(with_dots=False)

    # Extract only category codes
    category_codes = [
        code for code in all_codes if cm.is_category(code)
    ]

    # Build dataframe
    df = pd.DataFrame({"category_codes": category_codes})
    df["category_codes_des"] = df["category_codes"].apply(
        cm.get_description
    )

    # Get parent codes of category codes
    df = get_parent_vals(
        df, "category_codes", "block_codes", cm.is_block
    )

    # Get parent codes of block codes
    df = get_parent_vals(
        df, "block_codes", "chapter_codes", cm.is_chapter
    )

    # Save dataframe
    save_df(
        df,
        os.path.join(target_folder_path, "icd10_codes_and_des.csv"),
    )


@click.command()
@click.option(
    "--icd9",
    "-i9",
    help="Create ICD9 codes and descriptions",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--icd10",
    "-i10",
    help="Create ICD10 codes and descriptions",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--all",
    "-a",
    help="Create all codes and descriptions",
    default=False,
    is_flag=True,
    required=False,
)
def main(icd9: bool, icd10: bool, all: bool) -> None:
    """
    Make csv files with ICD codes and descriptions

    Parameters
    ----------
    icd9 : bool
        Flag to create ICD 9 codes and descriptions
    icd10 : bool
        Flag to create ICD 10 codes and descriptions
    all : bool
        Flag to create all codes and descriptions
    """

    if icd9 or all:
        print("Building ICD 9 dataframe...")
        make_icd_9_csv(
            os.path.join("data", "raw", "icd9"),
            os.path.join("data", "interim"),
        )

    if icd10 or all:
        print("Building ICD 10 dataframe...")
        make_icd_10_csv(
            os.path.join("data", "interim"),
        )


if __name__ == "__main__":
    main()
