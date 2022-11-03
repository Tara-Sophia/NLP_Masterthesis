from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-base")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-base"
)

tokenizer = processor.tokenizer
tokenizer.save_pretrained("tok/test/")

feature_extractor = processor.feature_extractor
feature_extractor.save_pretrained("fe/test/")
