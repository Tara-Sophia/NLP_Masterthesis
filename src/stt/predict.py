from transformers import AutoProcessor, AutoModelForCTC, pipeline

processor = AutoProcessor.from_pretrained(MODEL)

model = AutoModelForCTC.from_pretrained(MODEL)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0,
)


print(pipe(sample_2["audio"]["path"]))
