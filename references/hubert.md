# Facebook Hubert model

## Model description

The Hubert model was proposed in [Hubert: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden States](https://arxiv.org/abs/2106.07447) by Yannick Jadoul, Abdelrahman Mohamed, Michael Auli, Sergey Edunov, Myle Ott, Naman Goyal, Jiatao Gu, Sergey Zagoruyko, Lukasz Kaiser, Vishrav Chaudhary, Piotr Żelasko, Edouard Grave, Armand Joulin, Michael Matena, Yanqi Zhou, Patrick von Platen, Francisco Massa, Gabriel Synnaeve, Abdelrahman Mohamed, Michael Auli. Hubert is a self-supervised learning method for pre-training speech representations that is trained on unlabeled speech data. The model is trained to predict the next token in a sequence of feature vectors extracted from the audio input. Hubert is trained using a contrastive learning objective and is able to learn representations that are invariant to common audio distortions. The model can be fine-tuned on downstream tasks such as automatic speech recognition (ASR) and speaker recognition.

## Performance of the model

The Hubert model was able to achieve a WER of 2.9% on the LibriSpeech test-clean dataset. The model was also able to achieve a WER of 7.5% on the LibriSpeech test-other dataset. The model was also able to achieve a WER of 3.5% on the Common Voice test dataset. The Hubert can be trained on just 1 GPU and can be trained in less than 24 hours.

### Perfromance of the wav2vec model compared to the Hubert model

The wav2vec model was able to achieve a WER of 3.5% on the LibriSpeech test-clean dataset. The model was also able to achieve a WER of 8.9% on the LibriSpeech test-other dataset. The model was also able to achieve a WER of 4.1% on the Common Voice test dataset.

# Key facts about Hubert

- Hubert is a self-supervised learning method for pre-training speech representations that is trained on unlabeled speech data.
- The model is trained to predict the next token in a sequence of feature vectors extracted from the audio input.
- Hubert is trained using a contrastive learning objective and is able to learn representations that are invariant to common audio distortions.
- The model can be fine-tuned on downstream tasks such as automatic speech recognition (ASR) and speaker recognition.
