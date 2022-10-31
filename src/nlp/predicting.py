from keybert import KeyBERT

# allow custom input sentence # input("Enter your text: ")
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

# from cleaning.py file import cleaning function
from cleaning import cleaning


def KeywordExtraction(text):
    # load our trained model from models nlp semi supervised
    model = KeyBERT("models/nlp/semi_supervised/model")
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=10,
        top_n=5,
        use_mmr=True,
    )
    return keywords


def clean_input(text):
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
    text = input("Enter your syntomps: ")
    text = clean_input(text)
    keywords = KeywordExtraction(text)
    print(keywords)


# call script from terminal
if __name__ == "__main__":
    main()
