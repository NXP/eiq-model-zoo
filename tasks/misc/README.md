# Other models

## Datasets

This directory contains models that does not dall into vision or audio tasks.
The models featured in this directory are trained on the following datasets:

### 4 class motor imagery

The [4 class motor imagery dataset](https://www.bbci.de/competition/iv/desc_2a.pdf) consists of EEG data from 9
subjects. The dataset is a record of four different motor imagery tasks, namely the imagination of movement of the left
hand (class 1), right hand (class 2), both
feet (class 3), and tongue (class 4). Files are saved in the .mat format.

## Metrics

### Categorical Accuracy

Categorical accuracy is the number of correct predictions divided by the total number of predictions.

## Model List

 Model name                       | Architecture | Backbone | Training Dataset                    | Acc FP32 | Acc INT8 | Input size       | OPS  | Params | FP32 Size | INT8 Size | Compatibility 
----------------------------------|--------------|----------|-------------------------------------|----------|----------|------------------|------|--------|-----------|-----------|---------------
 [EegTCNet](./eegTCNet/README.md) | CNN          | custom   | Four class motor imagery (001-2014) | 0.281    | 0.284    | (1, 1, 22, 1125) | 14 M | 4096   | 32 KB     | 28 KB     | i.MX 8MP, i.MX 93      

## References

[1] Thorir Mar Ingolfsson et al., "EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery
Brain-Machine Interfaces", 2020 arXiv preprint arXiv: [2006.00622](https://arxiv.org/pdf/2006.00622.pdf)

[2] Four class motor imagery (001-2014) http://bnci-horizon-2020.eu/database/data-sets

[3] Four class motor imagery, dataset 2a https://www.bbci.de/competition/iv/desc_2a.pdf