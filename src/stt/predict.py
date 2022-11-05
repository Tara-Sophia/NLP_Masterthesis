# -*- coding: utf-8 -*-
import io

import speech_recognition as sr
import torch
from decorators import log_function_name
from pydub import AudioSegment
from transformers import pipeline
from utils import (
    get_device,
    load_trained_model_and_processor_hubert,
    load_trained_model_and_processor_wav2vec2,
)


@log_function_name
def transcribe_audio(pipe, rec):
    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        audio = rec.listen(source, phrase_time_limit=2)
        print("Transcribing...")
        # Transcribe audio file
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data, format="wav")
        x = torch.FloatTensor(clip.get_array_of_samples()).numpy()
        text = pipe(x)["text"]
        print(text)


@log_function_name
def main():
    device = get_device()
    model_to_predict_with = "Hubert"  # "Wav2Vec2"
    print(f"Loading model: {model_to_predict_with}")
    if model_to_predict_with == "Hubert":
        model, processor = load_trained_model_and_processor_hubert(
            device
        )
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
    rec = sr.Recognizer()
    transcribe_audio(pipe, rec)


if __name__ == "__main__":
    main()
