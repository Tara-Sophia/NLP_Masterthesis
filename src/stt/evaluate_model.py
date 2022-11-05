# -*- coding: utf-8 -*-
"""
Description:
    Evaluate the trained model on the test set

Usage:
    $ python src/data/evaluate_model.py
"""
import os
import random
from typing import Union

import pandas as pd
import torch
from constants import PROCESSED_DIR
from datasets import Dataset
from datasets.arrow_dataset import Example
from decorators import log_function_name
from evaluate import EvaluationModule, load
from transformers import (
    HubertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from utils import (
    get_device,
    load_trained_model_and_processor_hubert,
    load_trained_model_and_processor_wav2vec2,
)


def map_to_result(
    batch: Example,
    model: Union[HubertForCTC, Wav2Vec2ForCTC],
    processor: Wav2Vec2Processor,
) -> Example:
    """
    Map batch to results for evaluation

    Parameters
    ----------
    batch : Example
        Batch of data
    model : Union[HubertForCTC, Wav2Vec2ForCTC]
        Trained model
    processor : Wav2Vec2Processor
        Processor used to transform data

    Returns
    -------
    Example
        Batch with results
    """
    with torch.no_grad():
        input_values = torch.tensor(
            batch["input_values"], device="cuda"
        ).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(
        batch["labels"], group_tokens=False
    )

    return batch


@log_function_name
def show_random_elements(
    dataset: Dataset, num_examples: int = 10
) -> None:
    """
    Show random predictions and labels from a dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset to show predictions and labels from
    num_examples : int, optional
        Number of predictions and labels to show, by default 10
    """
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)


@log_function_name
def showcase_test(
    model: Union[HubertForCTC, Wav2Vec2ForCTC],
    test_ds: Dataset,
    processor: Wav2Vec2Processor,
) -> None:
    """
    Showcase a test predictions with paddings

    Parameters
    ----------
    model : Union[HubertForCTC, Wav2Vec2ForCTC]
        Model to use for predictions
    test_ds : Dataset
        Test dataset
    processor : Wav2Vec2Processor
        Processor to decode predictions
    """
    with torch.no_grad():
        logits = model(
            torch.tensor(test_ds[:1]["input_values"], device="cuda")
        ).logits

    pred_ids = torch.argmax(logits, dim=-1)

    # convert ids to tokens
    converted_tokens = " ".join(
        processor.tokenizer.convert_ids_to_tokens(
            pred_ids[0].tolist()
        )
    )
    print(converted_tokens)


@log_function_name
def get_test_results(
    results: Dataset,
    wer_metric: EvaluationModule,
    cer_metric: EvaluationModule,
) -> None:
    """
    Print WER and CER of the test dataset

    Parameters
    ----------
    results : Dataset
        Test dataset with predictions and labels
    wer_metric : EvaluationModule
        WER metric
    cer_metric : EvaluationModule
        CER metric
    """
    print(
        "Test WER: {:.3f}".format(
            wer_metric.compute(
                predictions=results["pred_str"],
                references=results["text"],
            )
        )
    )
    print(
        "Test CER: {:.3f}".format(
            cer_metric.compute(
                predictions=results["pred_str"],
                references=results["text"],
            )
        )
    )


@log_function_name
def load_test_data(data_path: str) -> Dataset:
    """
    Load the test data

    Parameters
    ----------
    data_path : str
        Path to the test data

    Returns
    -------
    Dataset
        Test data
    """
    test_df = pd.read_feather(os.path.join(data_path, "test.feather"))
    test_ds = Dataset.from_pandas(test_df)
    return test_ds


@log_function_name
def main():
    """
    Main function
    """
    test_ds = load_test_data(PROCESSED_DIR)
    device = get_device()
    model_to_evaluate = "Hubert"  # "Wav2Vec2"
    print(f"Loading model: {model_to_evaluate}")
    if model_to_evaluate == "Hubert":
        model, processor = load_trained_model_and_processor_hubert(
            device
        )
    else:
        model, processor = load_trained_model_and_processor_wav2vec2(
            device
        )

    results = test_ds.map(
        map_to_result,
        fn_kwargs={"model": model, "processor": processor},
        remove_columns=test_ds.column_names,
    )

    wer_metric = load("wer")
    cer_metric = load("cer")

    get_test_results(results, wer_metric, cer_metric)

    show_random_elements(results)

    showcase_test(model, test_ds, processor)


if __name__ == "__main__":
    main()
