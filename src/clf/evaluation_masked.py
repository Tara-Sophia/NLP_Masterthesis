# -*- coding: utf-8 -*-

"""
Description:
    Comparing the performance of the different masked models
"""
import pickle

import numpy as np
import pandas as pd
from constants import (
    LR_MT_MASKED,
    RF_MT_MASKED,
    DT_MT_MASKED,
    SVM_MT_MASKED,
    XGB_MT_MASKED,
    TEST_DATA_DIR,
)
from imblearn.pipeline import Pipeline
from utils import load_data


# Other metrics for model evaluation (accuracy @k optimized for and MRR @k)
def reciprocal_rank(true_labels: list[str], machine_preds: list[str]) -> float:
    """
    Compute the reciprocal rank at cutoff k
    Parameters
    ----------
    true_labels : list[str]
        true labels
    machine_preds : list[str]
        machine predictions
    Returns
    -------
    float
        reciprocal rank
    """

    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_preds) if r in true_labels]

    rr = 0.0
    if len(tp_pos_list) > 0:
        # for RR we need position of first correct item
        first_pos_list = tp_pos_list[0]

        # rr = 1/rank
        rr = 1 / float(first_pos_list)

    return rr


def compute_mrr_at_k(items: list[list[list[str]]]) -> float:
    """
    Compute the MRR (average RR) at cutoff k
    Parameters
    ----------
    items : list[list[list[str]]]
        lists of lists (true labels, machine predictions)
    Returns
    -------
    float
        MRR @k
    """
    rr_total = 0.0
    mrr = 0.0

    for item in items:
        rr_at_k = reciprocal_rank(item[0], item[1])
        rr_total = rr_total + rr_at_k
        mrr = rr_total / 1 / float(len(items))

    return mrr


def compute_accuracy(eval_items: list[list[list[str]]]) -> float:
    """
    Compute the accuracy at cutoff k
    Parameters
    ----------
    eval_items :  list[list[list[str]]]
        list of tuples (true labels, machine predictions)
    Returns
    -------
    float
        accuracy @k
    """
    correct = 0

    for item in eval_items:
        true_pred = item[0]
        machine_pred = set(item[1])

        for cat in true_pred:
            if cat in machine_pred:
                correct += 1
                break

    accuracy = correct / float(len(eval_items))
    return accuracy


def collect_preds(Y_test: pd.Series, Y_preds: list[list[str]]) -> list[list[list[str]]]:
    """
    Collect all predictions and ground truth
    Parameters
    ----------
    Y_test : pd.Series
        true labels
    Y_preds : list[list[str]]
        list of machine predictions
    Returns
    -------
     list[list[list[str]]]
        lists of lists (true labels, machine predictions)
    """

    pred_gold_list = [[[Y_test.iloc[idx]], pred] for idx, pred in enumerate(Y_preds)]
    return pred_gold_list


def get_top_k_predictions(
    model: Pipeline,
    X_test: pd.Series,
    k: int,
) -> list[list[str]]:
    """
    Get top k predictions for each test sample
    Parameters
    ----------
    model : Pipeline
        model
    X_test : pd.Series
        test data
    k : int
        number of predictions
    Returns
    -------
    list[list[str]]
        list of top k predictions
    """

    probs = model.predict_proba(X_test)
    best_n = np.argsort(probs, axis=1)[:, -k:]
    preds = [
        [model.classes_[predicted_cat] for predicted_cat in prediction]
        for prediction in best_n
    ]

    preds = [item[::-1] for item in preds]
    return preds


def main():
    """
    Main function
    """
    # Load test data
    X_test, y_test = load_data(TEST_DATA_DIR)

    # Load models
    lr_mt_masked = pickle.load(open(LR_MT_MASKED, "rb"))

    rf_mt_masked = pickle.load(open(RF_MT_MASKED, "rb"))

    dt_mt_masked = pickle.load(open(DT_MT_MASKED, "rb"))

    svm_mt_masked = pickle.load(open(SVM_MT_MASKED, "rb"))

    xgb_mt_masked = pickle.load(open(XGB_MT_MASKED, "rb"))

    # evaluate model and print results for each model in dataframe
    models = [
        ("Logistic Regression (masked)", lr_mt_masked),
        ("Random Forest (masked)", rf_mt_masked),
        ("Decision Tree (masked)", dt_mt_masked),
        ("SVM (masked)", svm_mt_masked),
        ("XGBoost (masked)", xgb_mt_masked),
    ]

    results = []
    for name, model in models:
        preds = get_top_k_predictions(model, X_test, 3)
        eval_items = collect_preds(y_test, preds)
        accuracy = compute_accuracy(eval_items)
        mrr = compute_mrr_at_k(eval_items)
        results.append((name, accuracy, mrr))

    df = pd.DataFrame(results, columns=["Model", "Accuracy@3", "MRR@3"])
    print(df)


if __name__ == "__main__":
    main()
