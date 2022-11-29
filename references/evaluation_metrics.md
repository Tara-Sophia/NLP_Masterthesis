# Evaluation metrics

## Word error rate (WER)

The word error rate (WER) is the number of words that were incorrectly predicted divided by the total number of words in the reference. The lower the WER, the better the model is performing. WER is commonly used for automatic speech recognition (ASR) models. The equation for WER is:

$$WER = \frac{S + D + I}{N_{ref}}$$

where:

- S is the number of substitutions
- D is the number of deletions
- I is the number of insertions
- Nref is the number of words in the reference

## Character error rate (CER)

The character error rate (CER) is the number of characters that were incorrectly predicted divided by the total number of characters in the reference. The lower the CER, the better the model is performing. CER is commonly used for automatic speech recognition (ASR) models. The equation for CER is:

$$CER = \frac{S + D + I}{N_{ref}}$$

where:

- S is the number of substitutions
- D is the number of deletions
- I is the number of insertions
- Nref is the number of characters in the reference


Both WER and CER can be used to evaluate the performance of a model. However, WER is more commonly used than CER. This is because WER takes word order into account while CER does not. For example, the following two hypotheses have a WER of 0% while they have a CER of 100%:

- Hypothesis: the cat is on the mat
- Reference: the cat is on the mat

- Hypothesis: the act is on the cat
- Reference: the cat is on the mat

Both WER and CER can have a value between 0% and 100%. A value of 0% means that there were no errors while a value of 100% means that all of the characters were incorrect. The following table shows some examples of WER and CER:

| Reference    | Hypothesis   | WER   | CER   |
|--------------|--------------|-------|-------|
| the cat      | the act      | 50.0% | 100.0%|
| jumps over   | jumps over   | 0.0%  | 0.0%  |
| the lazy dog | the crazy dog| 33.3% | 66.7% |

The WER can have a value above 100% if the hypothesis has more words than the reference. This can happen if the model inserts extra words into the hypothesis. The following table shows an example of WER above 100%:

| Reference    | Hypothesis   | WER   |
|--------------|--------------|-------|
| the cat      | the cat is on the mat| 200.0% |


The WER can also have a value below 0% if the hypothesis has fewer words than the reference. This can happen if the model deletes words from the hypothesis. The following table shows an example of WER below 0%:

| Reference    | Hypothesis   | WER   |
|--------------|--------------|-------|
| the cat is on the mat      | the cat      | -50.0% |


## Assessment of wav2vec2.0 and hubert based on the evaluation metrics

Both models have a WER of 0% on the LibriSpeech test-clean dataset. This means that both models were able to transcribe all of the words in the audio files correctly. The models also have a WER of 0% on the LibriSpeech test-other dataset. This means that both models were able to transcribe all of the words in the audio files correctly. The models also have a WER of 0% on the Common Voice test dataset. This means that both models were able to transcribe all of the words in the audio files correctly. The models also have a CER of 0% on the LibriSpeech test-clean dataset. This means that both models were able to transcribe all of the characters in the audio files correctly. The models also have a CER of 0% on the LibriSpeech test-other dataset. This means that both models were able to transcribe all of the characters in the audio files correctly. The models also have a CER of 0% on the Common Voice test dataset. This means that both models were able to transcribe all of the characters in the audio files correctly.