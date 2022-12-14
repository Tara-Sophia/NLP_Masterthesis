"""
Description:
    This script is used to extract keywords from the medical transcription
"""

import pandas as pd
from keybert import KeyBERT
from transformers import AutoTokenizer, pipeline

from src.nlp.constants import (
    # MODEL_MLM_DIR,
    MTSAMPLES_TC_DIR,
    MTSAMPLES_PROCESSED_PATH_DIR,
    MODEL_TC_MT_DIR,
)


def keyword_extraction(x: str, model, nr_candidates: int, top_n: int) -> list[tuple]:
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
    tokenizer = AutoTokenizer.from_pretrained(model, model_max_lenght=512)

    # Truncate all the text to 512 tokens

    hf_model = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer,
    )

    kw_model = KeyBERT(model=hf_model)
    keywords = kw_model.extract_keywords(
        x,
        keyphrase_ngram_range=(1, 1),
        # ngram means the number of words in a keyword (1 means 1 word, 2 means 2 words). we used 1 because we want to extract single words the reasin is
        stop_words="english",
        use_maxsum=True,
        nr_candidates=nr_candidates,
        top_n=max(top_n),
        use_mmr=True,
        diversity=0.5,
        # diversity means that the keywords should be different from each other. The higher the diversity, the more different the keywords are. The default value is 0.7.
        # we used 0.5 because we want to have more keywords that are similar to each other  (e.g. "heart failure" and "heart failure symptoms")
    )
    return keywords


# n_gram stands for     n-gram means the number of words in a keyword (1 means 1 word, 2 means 2 words). we used 1 because we want to extract single words the reasin is

# extract keywords from transcription column and create new column with keywords
def keywords_from_model(
    df: pd.DataFrame, model: str, input_column_name: str, output_column_name: str
) -> pd.DataFrame:
    """
    Extract keywords from the input text using the TC model
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the medical transcription text to extract keywords from
    model : str
        Path to the model to use for keyword extraction
    input_column_name : str
        Name of final column with keywords
    Returns
    -------
    pd.DataFrame
        Dataframe with the keywords and weights extracted from the input text
    """
    # use input column name for column reference
    weights_name = f"{output_column_name}_weights"
    df[weights_name] = keyword_extraction(
        df[input_column_name], model, df["nr_candidates"], df["top_n"]
    )
    df[weights_name] = df.apply(lambda x: x[weights_name][: x["top_n"]], axis=1)
    df[output_column_name] = df[weights_name].apply(lambda x: [i[0] for i in x])
    return df


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save dataframe to csv file
    Parameters
    ----------
    df : pd.DataFrame
        This is the final dataframe with the keywords and weights to save
    """
    df.to_csv(path, index=False)


# make df column smaller than 512
def small_column_df(df: pd.DataFrame, column) -> pd.DataFrame:
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
    df[column] = df[column].str[:512]
    return df


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
    nr_candidates = int(nr_words * 40 / 100)
    if nr_candidates > 35:
        nr_candidates = 35
    return nr_candidates


def main() -> None:
    """
    Main function to run the script
    """
    df_large_column = pd.read_csv(MTSAMPLES_PROCESSED_PATH_DIR)

    # apply function to make column smaller than 512
    df = small_column_df(df_large_column, "transcription")

    df["nr_candidates"] = df["transcription"].apply(calculate_optimal_candidate_nr)

    # Top n keywords to extract
    df["top_n"] = df["nr_candidates"].apply(lambda x: round(x * 0.7))
    df_tc = keywords_from_model(
        df, MODEL_TC_MT_DIR, "transcription", "transcription_f_TC"
    )

    save_dataframe(df_tc, MTSAMPLES_TC_DIR)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":

    main()
