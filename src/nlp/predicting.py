# -*- coding: utf-8 -*-
"""
Description:
    This script is used for predicting on custom input
"""
from src.nlp.constants import MODEL_MLM_DIR
from src.nlp.keybert_mtsamples import keyword_extraction
from src.nlp.utils import cleaning_input


# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("stopwords")
# nltk.download("punkt")
# check that text is max 512 tokens long otherwise make it max 512 tokens long
def max_length(x: str) -> str:
    """
    This function checks if the input text is longer than 512 tokens.
    If it is, it truncates it to 512 tokens.

    Parameters
    ----------
    x : str
        Input text.

    Returns
    -------
    str
        Truncated text.
    """
    if len(x.split()) > 512:
        x = " ".join(x.split()[:512])
    return x


def main() -> None:
    """
    Main function
    """
    text = input("Enter your syntomps: ")
    text = cleaning_input(text)
    text = max_length(text)
    model = MODEL_MLM_DIR

    keywords = keyword_extraction(text, model)
    keywords_without_weight = [keyword[0] for keyword in keywords]
    print(keywords)
    print(keywords_without_weight)


# call script from terminal
if __name__ == "__main__":
    main()
