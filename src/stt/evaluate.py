from transformers import AutoModelForCTC, Wav2Vec2Processor

processor = AutoProcessor.from_pretrained(MODEL)

model = AutoModelForCTC.from_pretrained(MODEL)

import torch


def map_to_result(batch):
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


results = timit["test"].map(
    map_to_result, remove_columns=timit["test"].column_names
)


print(
    "Test WER: {:.3f}".format(
        wer_metric.compute(
            predictions=results["pred_str"],
            references=results["text"],
        )
    )
)


show_random_elements(
    results.remove_columns(["speech", "sampling_rate"])
)


model.to("cuda")

with torch.no_grad():
    logits = model(
        torch.tensor(timit["test"][:1]["input_values"], device="cuda")
    ).logits

pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
converted_tokens = " ".join(
    processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())
)

print(converted_tokens)


# Load compute metric
# Load test data
# Create show_random_elements function


def load_test_data():
    test_ds = load_from_disk(os.path.join(DATA_PATH, "test"))
    return test_ds


def main():
    test_ds = load_test_data()
    results = test_ds.map(
        map_to_result, remove_columns=test_ds.column_names
    )


if __name__ == "__main__":
    main()
