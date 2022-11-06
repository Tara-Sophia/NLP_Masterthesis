from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords

# nltk.download("stopwords")
# nltk.download("punkt")

# nltk.download("wordnet")
# nltk.download("omw-1.4")
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import transformers
from transformers import pipeline

# apply model semi supervised to text
def KeywordExtraction(df):
    tokenizer = AutoTokenizer.from_pretrained(
        "models/nlp/semi_supervised/model", model_max_lenght=512
    )

    # truncate all the text to 512 tokens

    hf_model = pipeline(
        "feature-extraction",
        model="models/nlp/semi_supervised/model",
        tokenizer=tokenizer,  # "models/nlp/semi_supervised/model",
    )
    # tokenizer_kwargs = {
    #     "padding": True,
    #     "truncation": True,
    #     "max_length": 512,
    #     "return_tensors": "pt",
    # }

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


# def apply_keyword_on_Dataframe(df):
#     df["keywords_outcome_weights"] = df["transcription"].apply(
#         lambda x: KeywordExtraction(x)
#     )
#     # column only with keywords
#     df["transcription_f"] = df["keywords_outcome_weights"].apply(
#         lambda x: [item[0] for item in x]
#     )

#     return df


def save_dataframe(df):
    df.to_csv("data/processed/nlp/mtsamples/mtsamples_semisupervised.csv", index=False)


# make df column smaller than 512
def small_column_df(df):
    df["transcription"] = df["transcription"].str[:512]
    return df


def main():
    df_1 = pd.read_csv("data/processed/nlp/mtsamples/mtsamples_cleaned.csv")
    df = small_column_df(df_1)
    # apply keyword extraction on dataframe
    df_f = KeywordExtraction(df)
    # save dataframe
    save_dataframe(df_f)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()

import string
