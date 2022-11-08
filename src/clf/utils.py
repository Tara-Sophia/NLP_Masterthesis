# -*- coding: utf-8 -*-
"""
Description:
   Helper functions that are used in multiple places
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


def load_data(filepath) -> tuple[pd.Series, pd.Series]:
    """
    Load data from csv file

    Parameters
    ----------
    filepath : str
        path to csv file

    Returns
    -------
    tuple[pd.Series, pd.Series]
        tuple with X and y
    """
    df = pd.read_csv(filepath)
    X = df.keywords
    y = df.medical_specialty
    return X, y


def preprocessing_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline

    Returns
    -------
    Pipeline
        pipeline with preprocessing steps
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("smote", SMOTE(random_state=42)),
        ]
    )
    return pipeline
