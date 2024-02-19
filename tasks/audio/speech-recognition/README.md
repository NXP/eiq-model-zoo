# Speech Recognition

## Datasets

The Speech Recognition models featured in this Model Zoo are trained on the following datasets:

### Librispeech dataset

[Librispeech dataset](https://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. Files are saved in the .flac format and is released under a CC BY license. [2]

## Metrics

### Letter Error Rate

Letter Error Rate (LER) uses Levenshtein distance to compute difference between two strings. 

## Model List

 Model name                                  | Architecture | Backbone | Training Dataset | Acc FP32 | Acc INT8 | Input size   | OPS     | MParams | FP32 Size | INT8 Size | Compatibility     
---------------------------------------------|--------------|----------|------------------|----------|----------|--------------|---------|---------|-----------|-----------|-------------------
 [Wav2Letter](./micro-speech-LSTM/README.md) | CNN          | custom   | Librispeech      | N/A      | 0.0877 LER | (1, 296, 39) | 6 982 M | -       | N/A       | 23 MB     | i.MX 8MP, i.MX 93 

## References

[1] Collobert, Ronan, Christian Puhrsch, and Gabriel Synnaeve. "Wav2letter: an end-to-end convnet-based speech recognition system." arXiv preprint arXiv:1609.03193 (2016).

[2] Panayotov, Vassil, et al. "Librispeech: an asr corpus based on public domain audio books." 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015.
