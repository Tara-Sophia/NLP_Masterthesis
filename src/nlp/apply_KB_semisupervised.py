# -*- coding: utf-8 -*-
import string

import nltk
import pandas as pd
import transformers
from constants import (
    MODEL_UNSUPERVISED_MODEL_DIR,
    MTSAMPLES_FINAL,
    MTSAMPLES_PROCESSED_CLEANED_DIR,
    NLP_PROCESSED_PATH_DIR,
)
from keybert import KeyBERT

# nltk.download("wordnet")
# nltk.download("omw-1.4")
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer, pipeline

# nltk.download("stopwords")
# nltk.download("punkt")


# apply model semi supervised to text
def KeywordExtraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract keywords from text using KeyBERT. KeyBert uses the pretrained model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the medical transcription text to extract keywords from

    Returns
    -------
    pd.DataFrame
        Dataframe with the additional keywords column
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_UNSUPERVISED_MODEL_DIR, model_max_lenght=512
    )

    # truncate all the text to 512 tokens

    hf_model = pipeline(
        "feature-extraction",
        model=MODEL_UNSUPERVISED_MODEL_DIR,
        tokenizer=tokenizer,  # "models/nlp/semi_supervised/model",
    )

    kw_model = KeyBERT(model=hf_model)
    keywords = kw_model.extract_keywords(
        df["transcription"],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=10,
        top_n=5,
        use_mmr=True,
        diversity=0.5,
    )
    # put keywords in dataframe
    df["keywords_outcome_weights"] = keywords

    return df


def save_dataframe(df: pd.DataFrame) -> None:
    """
    Save dataframe to csv file

    Parameters
    ----------
    df : pd.DataFrame
        This is the final dataframe with the keywords and weights to save
    """
    df.to_csv(MTSAMPLES_FINAL, index=False)


# make df column smaller than 512
def small_column_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the transcription column smaller than 512 tokens

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the medical transcription text to extract keywords from

    Returns
    -------
    pd.DataFrame
        Dataframe with the transcription column smaller than 512 tokens
    """
    df["transcription"] = df["transcription"].str[:512]
    return df


def main():
    """
    Main function to run the script
    """
    df_1 = pd.read_csv(MTSAMPLES_PROCESSED_CLEANED_DIR)
    df = small_column_df(df_1)
    # apply keyword extraction on dataframe
    df_f = KeywordExtraction(df)
    # save dataframe
    save_dataframe(df_f)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()
