def prepare_dataset(batch):
    # pre-process audio
    try:
        sample = batch[audio_column_name]
    except ValueError:
        # E22: some samples are empty (no audio). Reading the empty audio array will trigger
        # a soundfile ValueError. For now, we'll manually set these arrays to a zero array.
        # They will be filtered in the subsequent filtering stage and so are
        # explicitly ignored during training.
        sample = {
            "array": np.array([0.0]),
            "sampling_rate": sample_rate,
        }

    if resampler is not None:
        speech_tensor = torch.FloatTensor(sample["array"])
        speech_tensor = speech_tensor.squeeze()
        speech_tensor = resampler(speech_tensor)
        sample["array"] = speech_tensor.numpy()
        sample["sampling_rate"] = resampler.new_freq

    # For training Whisper we perform the audio preprocessing in the WhisperDataCollator
    # => we only need to supply it with the raw audio values
    batch["input_ids"] = sample["array"]
    batch["input_lengths"] = len(batch["input_ids"])

    # 'Error correction' of targets
    input_str = (
        batch[text_column_name].lower()
        if do_lower_case
        else batch[text_column_name]
    )

    # SPGISpeech
    if dataset_name == "kensho/spgispeech":
        pass  # no error correction necessary

    # JIWER compliance (for WER/CER calc.)
    # remove multiple spaces
    input_str = re.sub(r"\s\s+", " ", input_str)
    # strip trailing spaces
    input_str = input_str.strip()

    # Finally, we tokenize the processed text
    batch["labels"] = tokenizer(input_str).input_ids
    return batch


vectorized_datasets = raw_datasets.map(
    prepare_dataset,
    remove_columns=next(iter(raw_datasets.values())).column_names,
    num_proc=num_workers,
    desc="preprocess train dataset",
)


# filter training data with targets shorter than min_target_length or longer than max_target_length
def is_labels_in_length_range(labels):
    return min_target_length < len(labels) < max_target_length


if training_args.do_train:
    vectorized_datasets["train"] = vectorized_datasets[
        "train"
    ].filter(
        is_labels_in_length_range,
        num_proc=num_workers,
        input_columns=["labels"],
    )

# filter data with targets empty sentences
def is_labels_greater_than_min(labels):
    return len(labels) > 0


vectorized_datasets = vectorized_datasets.filter(
    is_labels_greater_than_min,
    num_proc=num_workers,
    input_columns=["labels"],
)
