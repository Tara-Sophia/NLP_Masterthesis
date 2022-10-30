from keybert import KeyBERT
import pandas as pd


def KeywordExtraction(model, text):
    model = KeyBERT(model_path="./results_modelBert")
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=25,
        top_n=10,
        use_mmr=True,
    )
    return keywords


def apply_keyword_on_Dataframe(model, df):
    df["keywords_outcome"] = df["transcription"].apply(
        lambda x: KeywordExtraction(model, x)
    )
    return df


def save_dataframe(df):
    df.to_csv("../data/raw/mtsamples_outcome_bert.csv", index=False)
    
    
def main():
    # data_path = "../data/raw/mtsamples_cleaned.csv"
    # dataset = load_dataset(data_path)
    # tokenized_dataset = tokenize_dataset(dataset)
    # tokenized_dataset = clean_remove_column(tokenized_dataset)
    # model = training_model(tokenized_dataset)
    df = pd.read_csv(data_path)
    df_outcome = apply_keyword_on_Dataframe(model, df)
    save_dataframe(df_outcome)


# Path: src/Keyword_Bert_Training.py
if __name__ == "__main__":
    main()