# Command Recognition

## Datasets

The Command Recognition models featured in this Model Zoo are trained on the following datasets:

### Speech commands dataset

[Speech commands dataset](https://arxiv.org/abs/1804.03209) is an audio dataset of spoken words designed to help train and evaluate keyword spotting systems. It has 65,000 one-second
long utterances of 30 short words, by thousands of different people. The dataset is designed to build basic but useful
voice interfaces for applications, with common words like “Yes”, “No”, digits, and directions included. Files are saved
in the .wav format and is released under a CC BY license. [1]

## Metrics

### Categorical Accuracy

Categorical accuracy is the number of correct predictions divided by the total number of predictions.

## Model List

 Model name                                        | Architecture | Backbone | Training Dataset | Acc FP32 | Acc INT8 | Input size      | OPS     | MParams | FP32 Size | INT8 Size | Compatibility               
---------------------------------------------------|--------------|----------|------------------|----------|----------|-----------------|---------|---------|-----------|-----------|-----------------------------
 [Microspeech LSTM](./micro-speech-LSTM/README.md) | RNN          | custom   | Speech commands  | 0.94     | 0.87     | (1, 49, 257)    | 0.023 M | 119923  | 122 KB    | 123 KB    | i.MX 8MP, i.MX 93           
 [DS CNN](./keyword-spotting_DSCNN/README.md)      | CNN          | custom   | Speech commands  | 0.95     | 0.92     | (1, 49, 10,  1) | 5M      | 23 756  | 43 KB     | 53 KB     | i.MX 8MP, i.MX 93, MCX N947 

## References

[1] Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition." arXiv preprint arXiv:
1804.03209 (2018).

[2] Banbury, Colby, et al. "Mlperf tiny benchmark." arXiv preprint arXiv:2106.07597 (2021).

[3] Microspeech LSTM Model
implementation: [Google colab](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/third_party/xtensa/examples/micro_speech_lstm/train/micro_speech_with_lstm_op.ipynb)