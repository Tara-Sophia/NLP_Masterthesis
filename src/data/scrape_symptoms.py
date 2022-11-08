# -*- coding: utf-8 -*-

"""
Description:
    Creating dataframes with ICD-9 and ICD-10 diseases.
    The data will be saved in data/interim folder.

Usage:
    $ python src/data/scrape_symptoms.py -a -s

Possible arguments:
    * -i9 or --icd9: ICD-9 codes
    * -i10 or --icd10: ICD-10 codes
    * -a or -all: ICD-9 and ICD-10 codes
    * -s or --save: Save dataframe to csv
"""

import sys
import json
import re
import unicodedata

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer

sys.path.insert(0, "..")
from decorators import log_function_name

WIKIPEDIA_BASE = "https://en.wikipedia.org/wiki"

NHS_BASE = "https://www.nhs.uk/conditions/"


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
    except (FileNotFoundError, ValueError):
        # If the file doesn't exist, create an empty dictionary
        cache = {}
    return cache


def get_symptoms_from_wiki(url: str) -> str | None:
    """
    Load symptoms from Wikipedia

    Parameters
    ----------
    url : str
        URL to wikipedia page

    Returns
    -------
    str | None
        Either a string with symptoms, Message with error with URL or None
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
            return "No Value"
    return None


def get_symptoms_from_nhs(url: str) -> str | None:
    """
    Load symptoms from NHS

    Parameters
    ----------
    url : str
        URL to NHS page

    Returns
    -------
    str | None
        Either a string with symptoms, Message with error or None
    """
    res = requests.get(url)
    if res.status_code in range(200, 300):
        soup = BeautifulSoup(
            res.content,
            "lxml",
            parse_only=SoupStrainer("section"),
        )
        for item in soup.find_all("h2"):
            if "Symptoms" not in item.text:
                continue
            else:
                try:
                    symptoms_list = item.find_next_siblings("ul")[0]
                    text = ", ".join(
                        [
                            li.text
                            for li in symptoms_list.find_all("li")
                        ]
                    )
                    text_cleaned = unicodedata.normalize("NFKD", text)
                    return text_cleaned
                except IndexError:
                    return "No Value"
        else:
            return "No Value"
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
    # Cleaning the description for URL
    code_des = (
        code_des.lower()
        .replace("other", "")
        .replace("diseases", "")
        .replace("infections", "")
        .replace(" ", "_")
        .strip("_ ")
    )
    if code_des not in symptoms_cache:
        symptoms_cache[code_des] = get_symptoms_from_wiki(
            f"{WIKIPEDIA_BASE}/{code_des}"
        )
        # If no symptoms are found on Wikipedia, then try NHS
        if symptoms_cache[code_des] == "No Value":
            symptoms_cache[code_des] = get_symptoms_from_nhs(
                f"{NHS_BASE}/{code_des.replace('_', '-')}"
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
    df.loc[:, "symptoms"] = df["category_codes_des"].apply(
        lambda x: get_symptoms(x, symptoms_cache)
    )
    return df


def save_data(
    file_path: str, df: pd.DataFrame, save: bool = False
) -> None:
    """
    Saving dataframe to csv

    Parameters
    ----------
    file_path : str
        Location to save dataframe
    df : pd.DataFrame
        Dataframe to save
    save : bool, optional
        Flag if dataframe should be saved, by default False
    """
    if save:
        df.to_csv(file_path, index=False)


def build_icd_symptoms_dataframe(
    df: pd.DataFrame, save: bool, name: str
) -> None:
    """
    Creating dataframe with ICD codes and symptoms

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with ICD codes
    save : bool
        Flag if dataframe should be saved
    name : str
        Name of file to save
    """
    # Load symptoms cache
    symptoms_cache = load_cache("./data/interim/symptoms.json")

    # Get sypmtoms from wikipedia
    df = create_symptoms_col(df, symptoms_cache)

    # Save dataframe
    save_data(f"./data/interim/{name}.csv", df, save)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loading dataframe

    Parameters
    ----------
    file_path : str
        Path todataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with ICD codes
    """
    df = pd.read_csv(file_path)
    return df


@click.command()
@click.option(
    "--icd9",
    "-i9",
    help="Append symptoms to ICD 9 dataframe",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--icd10",
    "-i10",
    help="Append symptoms to ICD 10 dataframe",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--all",
    "-a",
    help="Append symptoms to all dataframes",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--save",
    "-s",
    help="Save dataframe",
    default=False,
    is_flag=True,
    required=False,
)
def main(icd9: bool, icd10: bool, all: bool, save: bool) -> None:
    """
    Select which function to run

    Parameters
    ----------
    icd9 : bool
        Append symptoms to ICD 9 dataframe
    icd10 : bool
        Append symptoms to ICD 10 dataframe
    all : bool
        Append symptoms to all dataframes
    save : bool
        Flag to save dataframe
    """

    if icd9 or all:
        print("Scraping ICD 9 symptoms...")
        df = load_dataframe("./data/interim/icd9_codes_and_des.csv")
        build_icd_symptoms_dataframe(df, save, "icd9_symptoms")

    if icd10 or all:
        print("Scraping ICD 10 symptoms...")
        df = load_dataframe("./data/interim/icd10_codes_and_des.csv")
        build_icd_symptoms_dataframe(df, save, "icd10_symptoms")


if __name__ == "__main__":
    main()
