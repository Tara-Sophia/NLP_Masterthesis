# cleaning the dataframe
import pandas as pd
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def convert(lst_kw):
    """This function converts the keywords to a list of keywords"""
    list_keywords = lst_kw.split(",")
    return list_keywords


def location_indices(stringstext, check_list):
    """this function finds the location of the keywords in the string"""

    res = dict()
    for ele in check_list:
        if ele in stringstext:
            # getting front index
            strt = stringstext.index(ele)

            # getting ending index
            res[ele] = [strt, strt + len(ele) - 1]
    return res.values()


import string


def cleaning(sentence):
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


def main():
    data_path = "../data/raw/mtsamples.csv"
    df = pd.read_csv(data_path)
    # df['transcription'] = df['transcription'].tolist()
    df["transcription"] = df["transcription"].apply(cleaning)
    df["keywords"] = df["keywords"].fillna("")
    df["keywords_list"] = df["keywords"].apply(lambda x: x.split(","))
    df["location"] = df.apply(
        lambda x: location_indices(x.transcription, x.keywords_list), axis=1
    )
    df = df.dropna()
    # save dataframe
    df.to_csv("../data/processed/mtsamples_cleaned.csv", index=False)


if __name__ == "__main__":
    main()
