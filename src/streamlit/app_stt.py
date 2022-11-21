# -*- coding: utf-8 -*-
"""
Description:
    Implementation of the Streamlit app for the speech-to-text part of the project
"""
import torch
from audiorecorder import audiorecorder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline  # HubertForCTC,

import streamlit as st
from src.stt.constants import VOCAB_DIR, WAV2VEC2_MODEL_DIR


def get_device() -> torch.device:
    """
    Get torch device

    Returns
    -------
    torch.device
        Torch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]
        Trained wav2vec2 model and processor
    """
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(VOCAB_DIR)
    model.to(device)
    return model, processor


def stt_main() -> str:
    """
    Main function for the speech-to-text part of the project

    Returns
    -------
    str
        Transcription of the audio file
    """
    st.title("🎙️ Audio Recorder")
    audio = audiorecorder("Click to record", "Click to stop")

    device = get_device()
    model, processor = load_trained_model_and_processor_wav2vec2(device)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0,
    )

    if len(audio) > 0:
        audio_bytes = audio.tobytes()
        st.audio(audio_bytes)
        text = pipe(audio_bytes)
        st.subheader("Transcription")
        st.write("Text without spelling correction")
        st.write(text["text"])
        return text["text"]

    return "No transcription possible"
