# Anomaly Detection - Deep AutoEncoder

## Introduction

Anomalous sound detection (ASD) is the task to identify whether the sound emitted from a target machine is normal or
anomalous. This model was trained in an unsupervised way, which means that only normal sound samples have been provided
as training data.

The model is a part of [MLCommons tiny benchmark repository](https://github.com/mlcommons/tiny/tree/master) on
GitHub [1].

For the training authors used ToyADMOS dataset [2].

## Model Information

 Information          | Value                                                    
----------------------|----------------------------------------------------------
 Input shape          | (1, 640)                                                 
 Input example        | example_input.wav (Sound file from ToyADMOS dataset [2]) 
 Output shape         | (1, 640)                                                 
 Prediction           | float32 number                                           
 Prediction example   | 10.235820081893744                                       
 FLOPS                | 0.528 MOPS                                               
 Number of parameters | 267,928                                                  
 Source framework     | Tensorflow/Keras                                         
 Target platform      | MCU,MPU                                                  

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)), and on MCX
N947 using MCUXpresso SDK.

## Training and evaluation

The model has been trained and evaluated on the ToyADMOS dataset [2]. It achieved a score of 0.85 accuracy on the test
set according to the paper [1].

The original training procedure is
detailed [here](https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection).

## Conversion/Quantization

The model is downloaded directly into this folder and contains int8 quantized tflite model.
See [the source of the model](https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/02_convert.py)
for information on the quantization procedure that was used.

## Use case and limitations

This model can be used for anomaly detection applications. Since its accuracy is relatively low, it should not be
trusted for critical scenarios.

## Performance

Here are performance figures evaluated on i.MX 8MP using BSP LF6.1.1_1.0.0 and i.MX 93 using BSP LF6.1.36_2.1.0 and on
MCX N947 using MCUXpresso SDK, with
SDK version 2.13.0 MCXN10 PRC, Toolchain MCUXpresso IDE 11.7.1 and LibC NewlibNano (nohost):

 Model | Average latency | Platform     | Accelerator     | Command                                                                                                                                
-------|-----------------|--------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------
 Int8  | 0.152 ms        | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=ad01_int8.tflite                                                      
 Int8  | 0.151 ms        | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=ad01_int8.tflite  --num_threads=4                                     
 Int8  | 0.140 ms        | i.MX 8M Plus | NPU             | benchmark_model --graph=ad01_int8.tflite  --external_delegate_path=Libvx_delegate.so 
 Int8  | 0.152 ms        | i.MX 93      | CPU (1 thread)  | benchmark_model --graph=ad01_int8.tflite                                                      
 Int8  | 0.152 ms        | i.MX 93      | CPU (2 threads) | benchmark_model --graph=ad01_int8.tflite --num_threads=2                                      
 Int8  | 0.143 ms        | i.MX 93      | NPU             | benchmark_model --graph=ad01_int8.tflite --external_delegate_path=libvx_delegate.so  
 Int8  | 5.605 ms        | MCX N947     | CPU             | MCUXpresso SDK                                                                                                                         |
 Int8  | 0.797 ms        | MCX N947     | NPU             | MCUXpresso SDK                                                                                                                         


**Note**: Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf), to find out where benchmark_model, libvx_delegate and libethosu_delegate are located.

## Download and run

To create the TFLite model fully quantized in int8 with int8 input and float32 output, follow the top-level README instructions to install Docker and build the Docker image, then run the following command: 

    docker run --rm -v "$PWD:/workspace" nxp-model-zoo recipe.sh 

Original dataset will be downloaded too.

The TFLite model file for i.MX 8M Plus, i.MX 93 and MCX N947 is `ad01_int8.tflite`. 

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

An example of how to use the model is in `example.py`.

## Origin

Model implementation: https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection

[1] Colby Banbury et al. "MLPerf Tiny Benchmark", [arXiv:2106.07597](https://arxiv.org/abs/2106.07597), 2021

[2] Koizumi, Yuma, et al. "ToyADMOS: A dataset of miniature-machine operating sounds for anomalous sound detection."
2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2019..