import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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
