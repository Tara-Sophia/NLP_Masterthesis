# -*- coding: utf-8 -*-

import pandas as pd
from keybert import KeyBERT
from transformers import AutoTokenizer, pipeline


def KeywordExtraction(df):
    tokenizer = AutoTokenizer.from_pretrained(
        "models/nlp/unsupervised/model", model_max_lenght=512
    )

    # Truncate all the text to 512 tokens

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
    # Column only with keywords
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


# Make df column smaller than 512
def small_column_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncate the transcription column to 512 tokens

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with long medical transcription text

    Returns
    -------
    pd.DataFrame
        Dataframe with the column truncated to 512 tokens
    """

    df["transcription"] = df["transcription"].str[:512]
    return df


def main() -> None:
    """
    Main function
    """
    df_large_column = pd.read_csv(
        "data/processed/nlp/mtsamples/mtsamples_semisupervised.csv"
    )
    df = small_column_df(df_large_column)
    df = KeywordExtraction(df)

    # Apply keyword extraction on dataframe
    df = apply_keyword_on_Dataframe(df)

    # Save dataframe
    save_dataframe(df)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()
