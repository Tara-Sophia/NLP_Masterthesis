# -*- coding: utf-8 -*-
import json
import re
from typing import Callable

import click
import pandas as pd
import requests
import simple_icd_10_cm as cm
from bs4 import BeautifulSoup, SoupStrainer

BASE = "https://en.wikipedia.org/wiki"


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


def load_cache(file_path: str) -> dict:
    """
    Loading cache with already scraped symptoms

    Parameters
    ----------
    file_path : str
        Path to location of cache file

    Returns
    -------
    dict
        Either empty dictionary or dictionary with symptoms
    """
    try:
        with open(file_path) as file:
            # Load its content as a dictionary
            cache = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        cache = {}
    return cache


def get_symptoms_from_wiki(url: str) -> str | None:
    """
    Load symptoms from wikipedia

    Parameters
    ----------
    url : str
        URL to wikipedia page

    Returns
    -------
    str | None
        Either a string with symptoms, IndexError with URL or None
    """
    res = requests.get(url)
    if res.status_code in range(200, 300):
        soup = BeautifulSoup(
            res.content,
            "lxml",
            parse_only=SoupStrainer("table", class_="infobox"),
        )
        try:
            text = soup.findAll("th", string="Symptoms")[
                0
            ].next_sibling.text
            text_cleaned = re.sub(r"[\(\[].*?[\)\]]", "", text)
            return text_cleaned
        except IndexError:
            return f"IndexError: {url}"
    return None


def get_symptoms(code_des: str, symptoms_cache: dict) -> str | None:
    """
    Return sypmtoms of ICD code

    Parameters
    ----------
    code_des : str
        Description of ICD code
    symptoms_cache : dict
        Cache of already scraped symptoms

    Returns
    -------
    str | None
        Either list os symptoms or None
    """
    code_des = code_des.replace(" ", "_")
    if code_des not in symptoms_cache:
        symptoms_cache[code_des] = get_symptoms_from_wiki(
            f"{BASE}/{code_des}"
        )

    with open("./data/interim/symptoms.json", "w") as fp:
        json.dump(symptoms_cache, fp)

    return symptoms_cache[code_des]


def create_symptoms_col(
    df: pd.DataFrame, symptoms_cache: dict
) -> pd.DataFrame:
    """
    Creation of symptoms column

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with ICD codes
    symptoms_cache : dict
        Cache of already scraped symptoms

    Returns
    -------
    pd.DataFrame
        Dataframe with symptoms column
    """
    df["symptoms"] = df["category_codes_des"].apply(
        lambda x: get_symptoms(x, symptoms_cache)
    )
    return df


def save_data(file_path: str, df: pd.DataFrame) -> None:
    """
    Saving dataframe to csv

    Parameters
    ----------
    file_path : str
        Location to save dataframe
    df : pd.DataFrame
        Dataframe to save
    """
    df.to_csv(file_path, index=False)


def build_icd_symptoms_dataframe(count: int) -> None:
    """
    Creating dataframe with ICD codes and symptoms
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

    # Load symptoms cache
    symptoms_cache = load_cache("./data/interim/symptoms.json")

    # Make dataframe smaller
    df = df.head(count)

    # Get sypmtoms from wikipedia
    df = create_symptoms_col(df, symptoms_cache)

    # Save dataframe
    save_data("./data/interim/icd10_symptoms.csv", df)


@click.command()
@click.option(
    "--icd_symptoms_dataframe",
    "-i",
    help="Build dataframe with ICD codes and symptoms",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--count",
    "-c",
    help="Sample size",
    default=10,
    required=True,
    type=int,
)
def main(icd_symptoms_dataframe: bool, count: int) -> None:
    """
    Select which function to run

    Parameters
    ----------
    icd_symptoms_dataframe : bool
        Create dataframe with ICD codes and symptoms
    count : int
        Sample size of dataframe
    """

    if icd_symptoms_dataframe:
        build_icd_symptoms_dataframe(count)


if __name__ == "__main__":
    main()

# TODO:
# https://www.cdc.gov/nocardiosis/symptoms/index.html
# if no symptoms found then go one level deeper
# If Indexerror then go to other website

# Update environment with: conda env update --file environment.yml
# The function can be run with the following command:
# python src/data/make_dataset.py -i -c 5
