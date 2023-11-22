# Anomaly detection

## Datasets

The anomaly detection models featured in this Model Zoo are trained on the following datasets:

### ToyAdmos

The [ToyAdmos dataset](https://github.com/YumaKoizumi/ToyADMOS-dataset) is a sound dataset of approximately 540 hours of
normal machine operating sounds and over 12,000 samples of anomalous sounds. It is designed for three tasks of ADMOS:
product inspection (toy car), fault diagnosis for fixed machine (toy conveyor), and fault diagnosis for moving machine (
toy train). The data are saved in the .wav format.

## Metrics

### Categorical Accuracy

Categorical accuracy is the number of correct predictions divided by the total number of predictions.

## Model List

 Model name                                       | Architecture | Backbone | Training Dataset | Acc FP32 | Acc INT8 | Input size | OPS     | Params  | FP32 Size | INT8 Size | Compatibility               
--------------------------------------------------|--------------|----------|------------------|----------|----------|------------|---------|---------|-----------|-----------|-----------------------------
 [Deep AutoEncoder](./deep-autoencoder/README.md) | AutoEncoder  | custom   | ToyADMOS         | 0.88     | 0.84     | (1, 640)   | 0.528 M | 267 928 | 1043 KB   | 271 KB    | i.MX 8MP, i.MX 93, MCX N947 

## References

[1] Colby Banbury et al. "MLPerf Tiny Benchmark", [arXiv:2106.07597](https://arxiv.org/abs/2106.07597), 2021

[2] Koizumi, Yuma, et al. "ToyADMOS: A dataset of miniature-machine operating sounds for anomalous sound detection."
2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA).