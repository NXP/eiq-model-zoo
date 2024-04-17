# EEG TCNet

## Introduction

This model is Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces. Its low memory
footprint and low computational complexity for inference make it suitable for embedded classification on
resource-limited
devices at the edge [1]. It is trained and tested on BCI Competition IV-2a dataset [3].

Model implementation is available on [GitHub](https://github.com/iis-eth-zurich/eeg-tcnet). Authors trained 9 models on
9 different datasets - subjects.

## Model Information

 Information          | Value                                                                                          
----------------------|------------------------------------------------------------------------------------------------
 Input shape          | Tabular data (1, 1, 22, 1125)                                                                  
 Output shape         | Vector of probabilities shape (1, 4). Labels are ['left hand', 'right hand', 'feet', 'tongue'] 
 Output example       | Output tensor: [[  87  -96 -120 -127]]                                                         
 FLOPS                | 14 MOPS                                                                                        
 Number of parameters | 4,096                                                                                          
 Source framework     | Tensorflow/Keras                                                                               

## Version and changelog

Initial release of quantized int8 model.

## Requirements

Python 3.7.4 is required to load original keras model.

Download here: https://www.python.org/downloads/release/python-374/

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Four class motor imagery (001-2014) dataset.

The original training procedure is detailed [here](https://github.com/iis-eth-zurich/eeg-tcnet).

## Conversion/Quantization

The model is downloaded in an archive and contains 9 different keras models (.pb). Quantization is done with
`quantize_model.py` script, located in this directory.

## Use case and limitations

The source repository provides experimental environment to reproduce results of the paper [1]. The model achieves 77.35%
classification accuracy on 4-class Motor Imagery dataset.

As of BSP LF6.1.36_2.1.0, model is not possible to accelerate on i.MX 8M Plus NPU. 

## Performance

Here are performance figures evaluated on i.MX 8MP and i.MX 93 using BSP LF6.1.36_2.1.0:

 Model | Average latency | Platform     | Accelerator     | Command                                                                                                                                         
-------|-----------------|--------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------
 Int8  | 23.0162 ms      | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=eegTCNet_quant_int8.tflite                                                     
 Int8  | 19.2728 ms      | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=eegTCNet_quant_int8.tflite --num_threads=4                                     
 Int8  | -               | i.MX 8M Plus | NPU             | benchmark_model --graph=eegTCNet_quant_int8.tflite --external_delegate_path=libvx_delegate.so 
 Int8  | 15.025 ms       | i.MX 93      | CPU (1 thread)  | benchmark_model --graph=eegTCNet_quant_int8.tflite                                                     
 Int8  | 13.884 ms       | i.MX 93      | CPU (2 threads) | benchmark_model --graph=eegTCNet_quant_int8.tflite --num_threads=2                                                    
 Int8  | 15.355 ms       | i.MX 93      | NPU             | benchmark_model --graph=eegTCNet_quant_int8.tflite --external_delegate_path=libethosu_delegate.so                                                    

**Note**: Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf), to find out where benchmark_model, libvx_delegate and libethosu_delegate are located.

## Download and run

0. Download and install Python 3.7.4 : https://www.python.org/downloads/release/python-374/
1. run ```bash recipe.sh```recipe.sh to download required files, prepare dataset and quantize model

The TFLite model file for i.MX 8M Plus and i.MX 93 is `.tflite`. 

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

An example of how to use the model is in `example.py`. Labels are: ['left hand', 'right hand', 'feet', 'tongue']

In case, one would like to use own dataset, the description of the original
dataset can to be checked [here](https://www.bbci.de/competition/iv/desc_2a.pdf).
Run ``` python prepare_dataset --path 'data_path'``` to
prepare binary files for inference.

## Origin

Model implementation:  https://github.com/iis-eth-zurich/eeg-tcnet

[1] Thorir Mar Ingolfsson et al., "EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery
Brain-Machine Interfaces", 2020 arXiv preprint arXiv: [2006.00622](https://arxiv.org/pdf/2006.00622.pdf)

[2] Four class motor imagery (001-2014) https://bnci-horizon-2020.eu/database/data-sets

[3] Four class motor imagery, dataset 2a https://www.bbci.de/competition/iv/desc_2a.pdf
