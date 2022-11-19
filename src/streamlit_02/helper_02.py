# -*- coding: utf-8 -*-
# keybert predicting
# Path: src/predicting.py
# Compare this snippet from src/training.py:
#         checkpoint, num_labels=38

# import pretrained model for Keybert

# allow custom input sentence # input("Enter your text: ")
"""
Description:
    This is a helper function to extract keywords
    from text using KeyBERT
"""

from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModel


def KeywordExtraction(text: str) -> list[str]:
    """
    Extract keywords from text using KeyBERT

    Parameters
    ----------
    text : str
        Text to extract keywords from

    Returns
    -------
    list[str]
        List of keywords
    """
    # model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # path might not be right after spliiting the PR
    model = AutoModel.from_pretrained("../../../models/mtsamples/semi_supervised")
    model = KeyBERT(model=model)
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=15,
        top_n=10,
        use_mmr=True,
    )
    return keywords


def clean_input(text: str) -> str:
    """
    Clean the input text

    Parameters
    ----------
    text : str
        Text to clean

    Returns
    -------
    str
        Cleaned text
    """
    lemmatizer = WordNetLemmatizer()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("  ", " ")
    # remove stopwords and numbers and punctuation
    text = " ".join(  # join the list of words into a string
        [
            word
            for word in text.split()
            if word not in stopwords.words("english")
            # remove numbers and punctuation
            and word.isalpha()
        ]
    )
    # lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


def main():
    """
    Main function
    """
    text = input("Enter your syntomps: ")
    text = clean_input(text)
    keywords = KeywordExtraction(text)
    print(keywords)


# call script from terminal
if __name__ == "__main__":
    main()
