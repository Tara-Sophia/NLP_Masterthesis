# -*- coding: utf-8 -*-
import os
import random

import pandas as pd
import torch
from constants import DATA_PATH_DATASETS, MODEL_DIR
from datasets import load_from_disk
from decorators import log_function_name
from evaluate import load
from transformers import AutoProcessor, Wav2Vec2ForCTC
from wav2vec2 import get_device


@log_function_name
def map_to_result(batch, model, processor):
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
def show_random_elements(dataset, num_examples=10):
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
def showcase_test(model, test_ds, processor):
    # Show the actual transcription as well
    # Make it random
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
def load_test_data(data_path):
    test_ds = load_from_disk(os.path.join(data_path, "val"))
    return test_ds


@log_function_name
def load_model_and_processor(device):
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model.to(device)
    return model, processor


@log_function_name
def main():
    test_ds = load_test_data(DATA_PATH_DATASETS)
    test_ds = test_ds.select(range(10))
    device = get_device()
    model, processor = load_model_and_processor(device)
    print("Loading processor")

    results = test_ds.map(
        map_to_result,
        fn_kwargs={"model": model, "processor": processor},
        remove_columns=test_ds.column_names,
    )

    wer_metric = load("wer")

    print(
        "Test WER: {:.3f}".format(
            wer_metric.compute(
                predictions=results["pred_str"],
                references=results["text"],
            )
        )
    )

    show_random_elements(results)

    showcase_test(model, test_ds, processor)


if __name__ == "__main__":
    main()
