# -*- coding: utf-8 -*-
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./models/stt/openai/tokenizer/"
)
print(tokenizer)


import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
)
sample = dataset[0]

# load model and processor
model_o = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base.en"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-base.en",
    tokenizer=tokenizer,
)

# # load dummy dataset and read soundfiles
input_speech = sample["audio"]["array"]
input_features = processor(
    input_speech, return_tensors="pt"
).input_features
predicted_ids = model_o.generate(input_features, max_length=448)
transcription = processor.batch_decode(
    predicted_ids, skip_special_tokens=True
)
print(transcription[0].strip())
