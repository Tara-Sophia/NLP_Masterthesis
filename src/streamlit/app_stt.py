# -*- coding: utf-8 -*-
"""
Description:
    Implementation of the Streamlit app for the speech-to-text part of the project
"""
from audiorecorder import audiorecorder
from transformers import AutoModelForCTC, AutoProcessor, pipeline

import streamlit as st


@st.cache(allow_output_mutation=True)
def load_pipeline() -> pipeline:
    """
    Load the pipeline

    Returns
    -------
    pipeline
        Pipeline
    """
    processor = AutoProcessor.from_pretrained("florentinhaugwitz/wav2vec2_medical")
    model = AutoModelForCTC.from_pretrained("florentinhaugwitz/wav2vec2_medical")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )
    return pipe


def stt_main() -> str | None:
    """
    Main function for the speech-to-text part of the app

    Returns
    -------
    str | None
        Transcription of the audio
    """
    st.header("🎙️ Audio Recorder")
    audio = audiorecorder("Click to record ▶️", "Click to stop ⏹")

    pipe = load_pipeline()

    if len(audio) > 0:
        audio_bytes = audio.tobytes()
        st.audio(audio_bytes)
        text = pipe(audio_bytes)
        with st.expander("Transcription"):
            st.write(text["text"])
        st.write("Thank you very much for your input!")
        st.write(
            "You can either listen to your audio, retake a new audio, or continue with the presentation of the most suitable medical departments."
        )
        return text["text"]

    return None
