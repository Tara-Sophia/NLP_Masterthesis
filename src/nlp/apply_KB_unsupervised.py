# -*- coding: utf-8 -*-
import string

import nltk
import pandas as pd
import transformers
from keybert import KeyBERT

# nltk.download("wordnet")
# nltk.download("omw-1.4")
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer, pipeline

# nltk.download("stopwords")
# nltk.download("punkt")


# create also column for keywords from semisupervised model


def KeywordExtraction(df):
    tokenizer = AutoTokenizer.from_pretrained(
        "models/nlp/unsupervised/model", model_max_lenght=512
    )

    # truncate all the text to 512 tokens

    hf_model = pipeline(
        "feature-extraction",
        model="models/nlp/unsupervised/model",
        tokenizer=tokenizer,  # "models/nlp/semi_supervised/model",
    )

    kw_model = KeyBERT(model=hf_model)
    keywords = kw_model.extract_keywords(
        df["transcription"],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=20,
        top_n=15,
        use_mmr=True,
        diversity=0.5,
    )
    df["keywords_outcome_weights_unsupervised"] = keywords

    return df


def apply_keyword_on_Dataframe(df):
    # column only with keywords
    df["transcription_f_unsupervised"] = df[
        "keywords_outcome_weights_unsupervised"
    ].apply(lambda x: [item[0] for item in x])

    df["transcription_f_semisupervised"] = df[
        "keywords_outcome_weights"
    ].apply(lambda x: [item[0] for item in x])
    return df


def save_dataframe(df):
    df.to_csv(
        "data/processed/nlp/mtsamples/mtsamples_unsupervised_both.csv",
        index=False,
    )


# make df column smaller than 512
def small_column_df(df):
    # take only the first 512 characters from column transcription

    df["transcription"] = df["transcription"].str[:512]
    return df


def main():
    df_1 = pd.read_csv(
        "data/processed/nlp/mtsamples/mtsamples_semisupervised.csv"
    )
    df = small_column_df(df_1)
    df = KeywordExtraction(df)
    # apply keyword extraction on dataframe
    df = apply_keyword_on_Dataframe(df)
    # save dataframe
    save_dataframe(df)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()
