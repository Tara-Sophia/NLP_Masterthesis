# Wav2vec2.0

## History

The wav2vec2.0 model succeeded the wav2vec model because it was able to achieve better results. The wav2vec model was trained on a large amount of data, but it was not able to achieve the same results as the wav2vec2.0 model. The difference between both models are 

## Model description

The wav2vec2.0 model was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli. Wav2Vec 2.0 is a self-supervised learning method for pre-training speech representations that is trained on unlabeled speech data. The model is trained to predict the next token in a sequence of feature vectors extracted from the audio input. Wav2Vec 2.0 is trained using a contrastive learning objective and is able to learn representations that are invariant to common audio distortions. The model can be fine-tuned on downstream tasks such as automatic speech recognition (ASR) and speaker recognition.

## Performance of the model

The wav2vec2.0 model was able to achieve a WER of 2.9% on the LibriSpeech test-clean dataset. The model was also able to achieve a WER of 7.5% on the LibriSpeech test-other dataset. The model was also able to achieve a WER of 3.5% on the Common Voice test dataset. The wav2vec2.0 can be trained on just 1 GPU and can be trained in less than 24 hours.

### Perfromance of the wav2vec model compared to the wav2vec2.0 model

The wav2vec model was able to achieve a WER of 3.5% on the LibriSpeech test-clean dataset. The model was also able to achieve a WER of 8.9% on the LibriSpeech test-other dataset. The model was also able to achieve a WER of 4.1% on the Common Voice test dataset.

# Key facts about Wave2Vec2.0

- Wav2Vec 2.0 is a self-supervised learning method for pre-training speech representations that is trained on unlabeled speech data.
- The model is trained to predict the next token in a sequence of feature vectors extracted from the audio input.
- Wav2Vec 2.0 is trained using a contrastive learning objective and is able to learn representations that are invariant to common audio distortions.
- The model can be fine-tuned on downstream tasks such as automatic speech recognition (ASR) and speaker recognition.


### Openai

The wav2vec and wav2vec2.0 models are both CNNs that learn representations of audio data in the frequency domain. The main difference between the two models is that wav2vec2.0 uses a deep CNN architecture while wav2vec uses a shallower CNN architecture. In addition, wav2vec2.0 uses a self-attention mechanism to better model long-term dependencies in the data.

Both wav2vec and wav2vec2.0 have been shown to outperform LSTM networks on a variety of tasks, including speech recognition and speaker identification. However, wav2vec2.0 generally outperforms wav2vec on these tasks.