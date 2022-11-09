# -*- coding: utf-8 -*-
"""
Description:
    Implementation of the Streamlit app for the speech-to-text part of the project

Usage:
    $ streamlit run src/data/streamlit_stt.py
"""
import torch
from audiorecorder import audiorecorder
from constants import HUBERT_MODEL_DIR, VOCAB_DIR, WAV2VEC2_MODEL_DIR
from transformers import (
    HubertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

import streamlit as st

torch.cuda.empty_cache()

CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:"\[\]]'


@st.experimental_memo
def get_device() -> torch.device:
    """
    Get torch device

    Returns
    -------
    torch.device
        Torch device
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# FACEBOOK WAV2VEC2
@st.experimental_memo
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
@st.experimental_memo
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


st.sidebar.title("Choose the model")
model_name = st.sidebar.selectbox("Model", ["Wav2Vec2", "Hubert"])
if model_name == "Wav2Vec2":
    model_to_load = WAV2VEC2_MODEL_DIR
else:
    model_to_load = HUBERT_MODEL_DIR


device = get_device()
if model_to_load == HUBERT_MODEL_DIR:
    model, processor = load_trained_model_and_processor_hubert(device)
else:
    model, processor = load_trained_model_and_processor_wav2vec2(
        device
    )
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0,
)

st.title("🎙️ Audio Recorder")
audio = audiorecorder("Click to record", "Click to stop")

if len(audio) > 0:
    audio_bytes = audio.tobytes()
    st.audio(audio_bytes)
    text = pipe(audio_bytes)
    st.subheader("Transcription")
    st.write("Text without spelling correction")
    st.write(text["text"])


st.header("Take audio from file")
audio_from_file = st.file_uploader("", type=[".wav"])

if audio_from_file:
    bytes_data = audio_from_file.getvalue()
    text = pipe(bytes_data)
    st.write(text)
