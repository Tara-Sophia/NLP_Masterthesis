import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from decorators import log_function_name
from constants import WAV2VEC2_MODEL_DIR


@log_function_name
def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@log_function_name
def load_model_and_processor(device):
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_DIR)
    model.to(device)
    return model, processor


@log_function_name
def load_processor(processor_path):

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        processor_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return processor
