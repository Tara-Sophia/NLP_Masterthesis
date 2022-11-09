# -*- coding: utf-8 -*-
"""
Description:
    Predict the transcription of an audio file

Usage:
    $ python src/data/predict.py -h or -w

Possible arguments:
    * -h or --hubert: Use Hubert model
    * -w or --wav2vec2: Use Wav2Vec2 model
"""
import io
import sys
import click

import speech_recognition as sr
import torch
from pydub import AudioSegment
from transformers import Pipeline, pipeline
from utils import (
    get_device,
    load_trained_model_and_processor_hubert,
    load_trained_model_and_processor_wav2vec2,
    correct_spelling,
)
from constants import SRC_DIR

sys.path.insert(0, SRC_DIR)
from decorators import log_function_name


@log_function_name
def transcribe_audio(pipe: Pipeline, rec: sr.Recognizer) -> None:
    """
    Transcribe audio from a microphone

    Parameters
    ----------
    pipe : Pipeline
        Speech recognition pipeline
    rec : sr.Recognizer
        Speech recognition recognizer
    """
    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        audio = rec.listen(source, phrase_time_limit=2)
        print("Transcribing...")
        # Transcribe audio file
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data, format="wav")
        x = torch.FloatTensor(clip.get_array_of_samples()).numpy()
        text = pipe(x)["text"]
        text_cleaned = correct_spelling(text)
        print(text_cleaned)


@click.command()
@click.option(
    "--hubert",
    "-h",
    help="Choose Hubert model",
    default=False,
    is_flag=True,
    required=False,
)
@click.option(
    "--wav2vec2",
    "-w",
    help="Choose Wav2vec2 model",
    default=False,
    is_flag=True,
    required=False,
)
@log_function_name
def main(hubert: bool, wav2vec2: bool) -> None:
    """
    Main function

    Parameters
    ----------
    hubert : bool
        Use Hubert model to predict
    wav2vec2 : bool
        Use Wav2vec2 model to predict
    """
    device = get_device()
    if hubert:
        print("Using model Hubert")
        model, processor = load_trained_model_and_processor_hubert(
            device
        )
    elif wav2vec2:
        print("Using model Wav2vec2")
        model, processor = load_trained_model_and_processor_wav2vec2(
            device
        )
    else:
        print("No model given as input")
        print("Please enter: python src/stt/predict.py -h or -w")
        sys.exit()

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
