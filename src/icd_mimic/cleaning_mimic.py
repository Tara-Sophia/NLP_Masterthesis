# -*- coding: utf-8 -*-
import re

import numpy as np
import pandas as pd
from constants import (
    MIMIC_DATA_CLEANED,
    MIMIC_DATA_DIR,
    personalized_stopwords_filtered,
)
from nltk.corpus import stopwords, words


def keep_only_text_from_choosen_headers(text, choosen_headers):
    # print(text)
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


def clean_text(text):
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
            if word not in personalized_stopwords_filtered
        ]
    )
    document = " ".join(
        [word for word in document.split() if len(word) > 1]
    )
    return document


def main():
    # import data
    df = pd.read_csv(MIMIC_DATA_DIR)
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
    # save to csv
    df.to_csv(MIMIC_DATA_CLEANED, index=False)
