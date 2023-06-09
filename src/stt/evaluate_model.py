# -*- coding: utf-8 -*-
"""
Description:
    Evaluate the trained model on the test set

Usage:
    $ python src/data/evaluate_model.py -h or -w

Possible arguments:
    * -h or --hubert: Use Hubert model
    * -w or --wav2vec2: Use Wav2Vec2 model
"""
import os
import random
import sys
from typing import Union

import click
import pandas as pd
import torch
from datasets import Dataset
from datasets.arrow_dataset import Example
from evaluate import EvaluationModule, load
from transformers import HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Processor

from src.decorators import log_function_name
from src.stt.constants import PROCESSED_DIR
from src.stt.utils import (
    get_device,
    load_trained_model_and_processor_hubert,
    load_trained_model_and_processor_wav2vec2,
)


def map_to_result(
    batch: Example,
    model: Union[HubertForCTC, Wav2Vec2ForCTC],
    processor: Wav2Vec2Processor,
    device: torch.device,
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
    device: torch.device
        Device to use for predictions

    Returns
    -------
    Example
        Batch with results
    """
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch


@log_function_name
def show_random_elements(dataset: Dataset, num_examples: int = 5) -> None:
    """
    Show random predictions and labels from a dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset to show predictions and labels from
    num_examples : int, optional
        Number of predictions and labels to show, by default 5
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
    pd.set_option("display.width", -1)
    print(
        df,
    )


@log_function_name
def showcase_test(
    model: Union[HubertForCTC, Wav2Vec2ForCTC],
    test_ds: Dataset,
    processor: Wav2Vec2Processor,
    device: torch.device,
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
    device: torch.device
        Device to use for predictions
    """
    with torch.no_grad():
        logits = model(torch.tensor(test_ds[:1]["input_values"], device=device)).logits

    pred_ids = torch.argmax(logits, dim=-1)

    # convert ids to tokens
    converted_tokens = " ".join(
        processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())
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


@click.command()
@click.option(
    "--hubert",
    "-h",
    help="Choose Hubert model",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--wav2vec2",
    "-w",
    help="Choose Wav2vec2 model",
    default=False,
    is_flag=True,
    required=False,
)
@log_function_name
def main(hubert: bool, wav2vec2: bool) -> None:
    """
    Main function

    Parameters
    ----------
    hubert : bool
        Use Hubert model to evaluate
    wav2vec2 : bool
        Use Wav2vec2 model to evaluate
    """
    test_ds = load_test_data(PROCESSED_DIR)
    device = get_device()

    # Decide which model to use
    if hubert:
        print("Using model Hubert")
        model, processor = load_trained_model_and_processor_hubert(device)
    elif wav2vec2:
        print("Using model Wav2vec2")
        model, processor = load_trained_model_and_processor_wav2vec2(device)
    else:
        print("No model given as input")
        print("Please enter: python src/stt/predict.py -h or -w")
        sys.exit()

    results = test_ds.map(
        map_to_result,
        fn_kwargs={
            "model": model,
            "processor": processor,
            "device": device,
        },
        remove_columns=test_ds.column_names,
    )

    # Calling the metrics
    wer_metric = load("wer")
    cer_metric = load("cer")

    get_test_results(results, wer_metric, cer_metric)

    show_random_elements(results)

    showcase_test(model, test_ds, processor, device)


if __name__ == "__main__":
    main()
