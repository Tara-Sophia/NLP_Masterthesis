import streamlit as st
import io
import numpy as np
from audiorecorder import audiorecorder
import torch
from constants import WAV2VEC2_MODEL_DIR, HUBERT_MODEL_DIR, VOCAB_DIR
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import speech_recognition as sr
from pydub import AudioSegment

torch.cuda.empty_cache()


@st.experimental_memo
def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@st.experimental_memo
def load_model_and_processor(device, model):
    model = Wav2Vec2ForCTC.from_pretrained(model)
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
model, processor = load_model_and_processor(device, model_to_load)
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
    st.write(text)
