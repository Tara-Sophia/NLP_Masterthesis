"""
Description:
   Helper functions that are used in multiple places
"""
import os
import pandas as pd
import imblearn
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    """
    Load data from csv file

    Parameters
    ----------
    filepath : str
        path to csv file

    Returns
    -------
    pd.DataFrame
        dataframe with data
    """
    df = pd.read_csv(filepath)
    X = df.keywords
    y = df.medical_specialty
    return X, y


def preprocessing_pipeline() -> imblearn.pipeline.Pipeline:
    """
    Create preprocessing pipeline

    Returns
    -------
    Pipeline
        pipeline with preprocessing steps
    """
    pipeline = imbPipeline(
        [
            ("vect", CountVectorizer()),
            ("smote", SMOTE(random_state=42)),
        ]
    )
    return pipeline
