# -*- coding: utf-8 -*-
import sounddevice as sd
import torch
from constants import WAV2VEC2_MODEL_DIR, VOCAB_DIR
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

import streamlit as st

torch.cuda.empty_cache()


@st.experimental_memo
def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording


@st.experimental_memo
def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@st.experimental_memo
def load_model_and_processor(device):
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(VOCAB_DIR)
    model.to(device)
    return model, processor


device = get_device()
model, processor = load_model_and_processor(device)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0,
)

# Implement start stop of recording
st.write("Record own voice")
if st.button("Click to Record"):
    record_state = st.text("Recording...")

    # Either this
    # fs = 16000
    # duration = 2
    # sd.default.samplerate = 16000
    # sd.default.channels = 1
    # myrecording = sd.rec(int(duration * fs))
    # sd.wait(duration)

    # Or this
    duration = 2  # seconds
    fs = 16000
    myrecording = record(duration=5, fs=48000)

    text = pipe(myrecording)
    st.write(text["text"])


st.write("Take audio from file")
audio = st.file_uploader("", type=[".wav"])

if audio:
    bytes_data = audio.getvalue()
    text = pipe(bytes_data)
    st.write(text)
