# -*- coding: utf-8 -*-
import io

import speech_recognition as sr
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
from utils import load_model_and_processor, get_device
from decorators import log_function_name


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
    model, processor = load_model_and_processor(device)
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
