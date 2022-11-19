# -*- coding: utf-8 -*-
import re

import pandas as pd
from nltk.corpus import stopwords

most_common_words = [
    "He",
    "She",
    "patient",
    "**]",
    "[**Hospital1",
    "The",
    "given",
    "showed",
    "also",
    "In",
    "On",
    "denies",
    "history",
    "found",
    "transferred",
    "ED",
    "Patient",
    "Name",
    "noted",
    "s/p",
    "started",
    "prior",
    "18**]",
    "admitted",
    "CT",
    "Pt",
    "2",
    "presented",
    "IV",
    "reports",
    "pt",
    "recent",
    "last",
    "received",
    "No",
    "BP",
    "ED,",
    "year",
    "old",
    "[**Known",
    "past",
    "1",
    "days",
    "lastname",
    "His",
    "OSH",
    "arrival",
    "time",
    "[**Last",
    "yo",
    "This",
    "presents",
    "well",
    "[**Hospital",
    "HR",
    "male",
    "mg",
    "x",
    "day",
    "Her",
    "admission",
    "without",
    "At",
    "home",
    "felt",
    "initial",
    "developed",
    "revealed",
    "(un)",
    "3",
    "since",
    "placed",
    "increased",
    "per",
    "A",
    "h/o",
    "recently",
    "CXR",
    "Per",
    "severe",
    "significant",
    "treated",
    "w/",
    "transfer",
    "L",
    "underwent",
    "initially",
    "[**Hospital3",
    "due",
    "states",
    "Denies",
    "one",
    "R",
    "notable",
    "symptoms",
    "seen",
    "ED.",
    "O2",
    "called",
    "RR",
    "status",
    "EKG",
    "several",
    "review",
    "Of",
    "feeling",
    "continued",
    "fevers,",
    "hospital",
    "[**Location",
    "(NI)",
    "Mr.",
    "went",
    "HTN,",
    "T",
    "(STitle)",
    "note,",
    "today",
    "VS",
    "became",
    "discharged",
    "MICU",
    "weeks",
    "ago",
    "episode",
    "4",
    "taken",
    "new",
    "sent",
    "normal",
    "[**Name",
    "medical",
    "episodes",
    "two",
    "chills,",
    "aortic",
    "100%",
    "denied",
    "improved",
    "possible",
    "unable",
    "SOB",
    "EMS",
    "morning",
    "associated",
    "elevated",
    "large",
    "reported",
    "brought",
    "week",
    "[**First",
    "RA.",
    "night",
    "course",
    "Dr.",
    "M",
    "GI",
    "decreased",
    "ICU",
    "WBC",
]


def keep_only_text_from_choosen_headers(
    text: str, choosen_headers: list
) -> str:
    """
    Keep only text from choosen headers

    Parameters
    ----------
    text : str
        Text to clean
    choosen_headers : list
        List of choosen headers

    Returns
    -------
    str
        Text with only choosen headers
    """
    new_text = ""
    # headers is a list of headers to keep

    text = str(text)
    # get all headers with regex (\n(.*?):\n)
    all_headers = re.findall(r"(\n(.*?):\n)", text)
    # get only first value in tuple
    all_headers = [x[1] for x in all_headers]

    # get the text after choosen headers until next header
    for c_h in choosen_headers:
        if c_h in text:
            if c_h in all_headers:
                # get index of choosen header
                index_c_h = all_headers.index(c_h)
                # get next header
                try:
                    next_header = all_headers[index_c_h + 1]
                    # get index of next header
                    index_next_header = text.index(next_header)
                    text_after_c_h = text[
                        text.index(c_h) + len(c_h) : index_next_header
                    ]
                    # add text to new text
                    new_text += text_after_c_h
                except IndexError:
                    print("IndexError")
                    # raise NameError('IndexError')

    return new_text


def clean_text(text: str) -> str:
    """
    Clean text

    Parameters
    ----------
    text : str
        Text to clean

    Returns
    -------
    str
       Cleaned text
    """

    # replace \n with space
    text = text.replace("\n", " ")
    # Remove all the special characters
    document = re.sub(r"\W", " ", str(text))
    # Remove all single characters
    document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)
    # Substituting multiple spaces with single space
    document = re.sub(r"\s+", " ", document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r"^b\s+", "", document)
    # Remove digits
    document = re.sub(r"\d+", "", document)
    # Remove everything in brackets
    document = re.sub(r"\[[^]]*\]", "", document)
    # Filter out stop words
    document = " ".join(
        [
            word
            for word in document.split()
            if word not in stopwords.words("english")
        ]
    )
    # Filter out most common words
    document = " ".join(
        [
            word
            for word in document.split()
            if word not in most_common_words
        ]
    )
    document = " ".join(
        [word for word in document.split() if len(word) > 1]
    )
    return document


def main():
    """
    Main function
    """
    df = pd.read_csv(
        "../../data/processed/mimic_iii/diagnoses_noteevents.csv"
    )
    # choosen headers
    df["TEXT_final"] = df["TEXT"].apply(
        lambda x: keep_only_text_from_choosen_headers(
            x,
            [
                "CHIEF COMPLAINT",
                "Chief Complaint",
                "Present Illness",
                "PRESENT ILLNESS",
                "History of Present Illness",
                "Impression",
                "ROS",
                "HISTORY OF PRESENT ILLNESS",
                "Respiratory",
            ],
        )
    )
    df["TEXT_final_cleaned"] = df["TEXT_final"].apply(clean_text)
