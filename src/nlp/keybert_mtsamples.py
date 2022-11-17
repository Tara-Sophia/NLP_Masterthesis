# -*- coding: utf-8 -*-
"""
Description:
    This script is used to extract keywords from the medical transcription
"""
# import lieral_eval
import pandas as pd
from constants import (
    MODEL_MLM_DIR,
    MODEL_TC_DIR,
    MTSAMPLES_FINAL,
    MTSAMPLES_PROCESSED_CLEANED_DIR,
)
from keybert import KeyBERT
from transformers import AutoTokenizer, pipeline


def keyword_extraction(
    x: str, model, nr_candidates: int, top_n: int
) -> list[tuple]:
    """
    This function extracts keywords from the input text.
    Parameters
    ----------
    x : str
        Input sentence.
    model : str
        Path to the model to use for keyword extraction
    Returns
    -------
    list[str]
        List of keywords.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model, model_max_lenght=512
    )

    # Truncate all the text to 512 tokens

    hf_model = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer,
    )

    kw_model = KeyBERT(model=hf_model)
    keywords = kw_model.extract_keywords(
        x,
        # df["transcription"],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=nr_candidates,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return keywords


def keywords_from_TC_model(
    df: pd.DataFrame, model: str
) -> pd.DataFrame:
    """
    Extract keywords from the input text using the TC model
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the medical transcription text to extract keywords from
    model : str
        Path to the model to use for keyword extraction
    Returns
    -------
    pd.DataFrame
        Dataframe with the keywords and weights extracted from the input text
    """

    df["keywords_outcome_weights_TC"] = df.apply(
        lambda x: keyword_extraction(
            x["transcription"], model, x["nr_candidates"], x["top_n"]
        ),
        axis=1,
    )

    df["transcription_f_TC"] = df[
        "keywords_outcome_weights_TC"
    ].apply(lambda x: [i[0] for i in x])
    return df


def keywords_from_MLM_model(
    df: pd.DataFrame, model: str
) -> pd.DataFrame:
    """
    Extract keywords from the input text using the MLM model
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the medical transcription text to extract keywords from
    model : str
        Path to the model to use for keyword extraction
    Returns
    -------
    pd.DataFrame
        Dataframe with the keywords and weights extracted from the input text
    """
    df["keywords_outcome_weights_MLM"] = df.apply(
        lambda x: keyword_extraction(
            x["transcription"], model, x["nr_candidates"], x["top_n"]
        ),
        axis=1,
    )

    df["transcription_f_MLM"] = df[
        "keywords_outcome_weights_MLM"
    ].apply(lambda x: [item[0] for item in x])
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


# calculate the optimal nr candidates for each individual text
def calculate_optimal_candidate_nr(text: str) -> int:
    """
    Calculate the optimal number of candidates for each text to use for
    keyword extraction
    Parameters
    ----------
    text : str
        Text to extract keywords from
    Returns
    -------
    int
        Optimal number of candidates to use for keyword extraction
    """
    nr_words = len(text.split())
    nr_candidates = int(nr_words * 20 / 100)
    if nr_candidates > 35:
        nr_candidates = 35
    return nr_candidates


def main() -> None:
    """
    Main function to run the script
    """
    df_large_column = pd.read_csv(MTSAMPLES_PROCESSED_CLEANED_DIR)
    df = small_column_df(df_large_column)
    df["nr_candidates"] = df["transcription"].apply(
        calculate_optimal_candidate_nr
    )
    # top n keywords to extract
    df["top_n"] = df["nr_candidates"].apply(lambda x: round(x * 0.5))
    # for MLM model :
    df_mlm = keywords_from_MLM_model(df, MODEL_MLM_DIR)
    # for TC model :
    df_tc = keywords_from_TC_model(df, MODEL_TC_DIR)
    df = pd.concat([df_mlm, df_tc], axis=1)
    save_dataframe(df)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()
