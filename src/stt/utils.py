# -*- coding: utf-8 -*-
"""
Description:
    Helper functions that are used in multiple places
"""
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    HubertForCTC,
    Trainer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from src.stt.constants import (
    HUBERT_MODEL_DIR,
    SAMPLING_RATE,
    VOCAB_DIR,
    WAV2VEC2_MODEL_DIR,
)
from src.stt.decorators import log_function_name


@log_function_name
def get_device() -> torch.device:
    """
    Get torch device

    Returns
    -------
    torch.device
        Torch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@log_function_name
def load_datasets(data_path: str) -> tuple[Dataset, Dataset]:
    """
    Load train and validation datasets

    Parameters
    ----------
    data_path : str
        Path to the data

    Returns
    -------
    tuple(Dataset, Dataset)
        Train and validation datasets
    """
    train_df = pd.read_feather(os.path.join(data_path, "train.feather"))
    val_df = pd.read_feather(os.path.join(data_path, "val.feather"))

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


@log_function_name
def load_processor(processor_path: str) -> Wav2Vec2Processor:
    """
    Load the processor with Tokenizer and Feature Extractor

    Parameters
    ----------
    processor_path : str
        Path to the processor

    Returns
    -------
    Wav2Vec2Processor
        Loaded processor
    """

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        processor_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor


# FACEBOOK WAV2VEC2
@log_function_name
def load_trained_model_and_processor_wav2vec2(
    device: torch.device,
) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    """
    Load the trained model and processor for wav2vec2

    Parameters
    ----------
    device : torch.device
        Torch device

    Returns
    -------
    tuple(Wav2Vec2ForCTC, Wav2Vec2Processor)
        Trained wav2vec2 model and processor
    """
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(VOCAB_DIR)
    model.to(device)
    return model, processor


# FACEBOOK HUBERT
@log_function_name
def load_trained_model_and_processor_hubert(
    device: torch.device,
) -> tuple[HubertForCTC, Wav2Vec2Processor]:
    """
    Load the trained model and processor for hubert

    Parameters
    ----------
    device : torch.device,
        Torch device

    Returns
    -------
    tuple(HubertForCTC, Wav2Vec2Processor)
        Trained hubert model and processor
    """
    model = HubertForCTC.from_pretrained(HUBERT_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(VOCAB_DIR)
    model.to(device)
    return model, processor


# TRAINING
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        """
        Constructs all the necessary attributes for the DataCollator object.

        Parameters
        ----------
        processor : Wav2Vec2Processor
            The processor used for proccessing the data.
        padding : Union[bool, str], optional
            If padding should be used, by default True
        max_length : Optional[int], optional
            Maximum length of padding, by default None
        max_length_labels : Optional[int], optional
            Maximum length of labels, by default None
        pad_to_multiple_of : Optional[int], optional
            Pad to multiple of, by default None
        pad_to_multiple_of_labels : Optional[int], optional
            Pad to multiple of labels, by default None
        """
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self,
        features: list[dict[str, Union[list[int], torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        Pad inputs and labels

        Parameters
        ----------
        features : list[dict[str, Union[list[int], torch.Tensor]]]
            List of inputs and labels

        Returns
        -------
        dict[str, torch.Tensor]
            Padded inputs and labels
        """
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


class EearlyStoppingCallbackAfterNumEpochs(EarlyStoppingCallback):
    """
    Callback to stop training after a number of epochs.
    Inherits from EarlyStoppingCallback.
    """

    def __init__(self, start_epoch: int, *args, **kwargs):
        """
        Constructs all the necessary attributes for the
        EearlyStoppingCallbackAfterNumEpochs object.

        Parameters
        ----------
        start_epoch : int
            Epoch to start the early stopping
        """
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs
    ) -> None:
        """
        Check if the training should be stopped.

        Parameters
        ----------
        args : TrainingArguments
            Training arguments
        state : TrainerState
            Trainer state
        control : TrainerControl
            Trainer control
        metrics : dict[str, float]
            Metrics to evaluate
        """
        if state.epoch > self.start_epoch:
            super().on_evaluate(args, state, control, metrics, **kwargs)


def compute_metrics(
    pred: EvalPrediction,
) -> dict[str, Union[float, str]]:
    """
    Compute metrics for the model during training

    Parameters
    ----------
    pred : EvalPrediction
        Predictions from the model

    Returns
    -------
    dict[str, Union[float, str]]
        Metrics to evaluate the performance of the model
    """
    processor = load_processor(VOCAB_DIR)
    cer_metric = load("cer")
    wer_metric = load("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "cer": cer,
        "wer": wer,
        "pred_str": pred_str[0],
        "label_str": label_str[0],
    }


@log_function_name
def load_training_args(
    output_dir: str,
    batch_size_train: int,
    batch_size_val: int,
    num_epochs: int,
) -> TrainingArguments:
    """
    Load the training arguments

    Parameters
    ----------
    output_dir : str
        Output directory for checkpoints
    batch_size_train : int
        Batch size for training
    batch_size_val : int
        Batch size for validation
    num_epochs : int
        Number of epochs to train

    Returns
    -------
    TrainingArguments
        Training arguments
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_val,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="wandb",
    )
    return training_args


@log_function_name
def load_trainer(
    model: Union[HubertForCTC, Wav2Vec2ForCTC],
    data_collator: DataCollatorCTCWithPadding,
    training_args: TrainingArguments,
    train_ds: Dataset,
    val_ds: Dataset,
    processor: Wav2Vec2Processor,
) -> Trainer:
    """
    Load the trainer for the model

    Parameters
    ----------
    model : Union[HubertForCTC, Wav2Vec2ForCTC]
        Model to train
    data_collator : DataCollatorCTCWithPadding
        Data collator for the model
    training_args : TrainingArguments
        Training arguments for the training process
    train_ds : Dataset
        Training dataset
    val_ds : Dataset
        Validation dataset
    processor : Wav2Vec2Processor
        Processor for the model

    Returns
    -------
    Trainer
        Trainer for the model
    """
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EearlyStoppingCallbackAfterNumEpochs(
                start_epoch=15,
                early_stopping_patience=5,
            )
        ],
    )
    return trainer
