# -*- coding: utf-8 -*-
# cleaning the dataframe
import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

nltk.download("wordnet")
nltk.download("omw-1.4")
from constants import (
    MTSAMPLES_PROCESSED_PATH_DIR,
    MTSAMPLES_RAW_PATH_DIR,
)
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# def convert(lst_kw):
#     """This function converts the keywords to a list of keywords"""
#     list_keywords = lst_kw.split(",")
#     return list_keywords


# def location_indices(stringstext, check_list):
#     """this function finds the location of the keywords in the string"""

#     res = dict()
#     for ele in check_list:
#         if ele in stringstext:
#             # getting front index
#             strt = stringstext.index(ele)

#             # getting ending index
#             res[ele] = [strt, strt + len(ele) - 1]
#     return res.values()


def cleaning_input(sentence):
    """This function takes a column and cleans it by removing punctuation, stopwords, and lemmatizing
    NEXT STEPS : personalize the stop words list, check for different aspects of the words
    """

    # Basic cleaning
    sentence = sentence.strip()  ## remove whitespaces
    sentence = sentence.lower()  ## lowercase
    sentence = "".join(
        char for char in sentence if not char.isdigit()
    )  ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")  ## remove punctuation

    tokenized_sentence = word_tokenize(sentence)  ## tokenize
    stop_words = set(stopwords.words("english"))  ## define stopwords

    tokenized_sentence_cleaned = [  ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words
    ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos="v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = " ".join(word for word in lemmatized)

    return cleaned_sentence


def create_df(df):
    df = df.dropna().copy()
    df["transcription"] = df["transcription"].apply(cleaning_input)

    df["keywords_list"] = df["keywords"].apply(lambda x: x.split(","))
    # df["location"] = df.apply(
    #     lambda x: location_indices(x.transcription, x.keywords_list), axis=1
    )
    return df


# filter only for classes with more than 100 samples
def top_11_classes(df):
    top_11 = df.value_counts("medical_specialty").index[:11]
    return df[df["medical_specialty"].isin(top_11)].copy()


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def save_df(df, path):
    create_dir(path)
    df.to_csv(os.path.join(path, "mtsamples_cleaned.csv"), index=False)


def main():
    df = pd.read_csv(os.path.join(MTSAMPLES_RAW_PATH_DIR, "mtsamples.csv"))
    df = create_df(df)
    print(df.shape)
    df = top_11_classes(df)
    # print size of the dataframe
    print(df.shape)

    # save dataframe
    save_df(df, MTSAMPLES_PROCESSED_PATH_DIR)


if __name__ == "__main__":
    main()
