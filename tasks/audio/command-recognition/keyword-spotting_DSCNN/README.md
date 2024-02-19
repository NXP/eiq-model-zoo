# Keyword spotting - DS CNN

Keyword spotting model is a speech recognition network, trained to recognize set of words (see [Labels](#labels)). It is
a part of the [MLCommons tiny benchmark repository on GitHub](https://github.com/mlcommons/tiny/tree/master) [2]. Model
architecture is depth-wise separable convolutional neural network, inspired by paper [3].

The model was trained on the Speech Commands dataset [1]. It is an audio dataset of spoken words, for training models
that detect when a single word is spoken. It’s released under
a [Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Introduction

## Model Information

 Information          | Value                                                                                                                                               
----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------
 Input shape          | Sound file ([ 1, 49, 10,  1])                                                                                                                       
 Input example        | 'example_input.wav',</br> Source: Speech commands dataset [1].</br> License [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/) 
 Output shape         | Vector of probabilities shape ([ 1, 12]). Labels can be found in [Labels](#labels).                                                                 
 Output example       | [0.99609375 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ]                                                                                                      
 FLOPS                | 5 MOPS                                                                                                                                              
 Number of parameters | 23 756                                                                                                                                              
 Source framework     | Tensorflow/Keras                                                                                                                                    
 Target platform      | MCU, MPU                                                                                                                                            

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Speech commands dataset. It demonstrated 91.6% accuracy, according to
the paper [2].

## Conversion/Quantization

The model is downloaded in an archive which contains the quantized model float and int TensorFlow model.
See [the source of the model](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/quantize.py)
for information on the quantization procedure that was used.

## Use case and limitations

This model can be used for speech command recognition applications. It was trained on the limited vocabulary (
see [Labels](#labels)), so its use case is also limited.

## Performance

Here are performance figures evaluated on i.MX 8MP using BSP LF6.1.1_1.0.0 and i.MX93 using BSP LF6.1.36_2.1.0 and on
MCX N947 using MCUXpresso SDK, with
SDK version 2.13.0 MCXN10 PRC, Toolchain MCUXpresso IDE 11.7.1 and LibC NewlibNano (nohost):

 Model | Average latency | Platform     | Accelerator     | Command                                                                                                                                   
-------|-----------------|--------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------
 Int8  | 0.938 ms        | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=kws_ref_model.tflite                                                     
 Int8  | 0.383 ms        | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=kws_ref_model.tflite --num_threads=4                                     
 Int8  | 0.182 ms        | i.MX 8M Plus | NPU             | benchmark_model --graph=kws_ref_model.tflite --external_delegate_path=libvx_delegate.so 
 Int8  | 0.440 ms        | i.MX 93      | CPU (1 thread)  | benchmark_model --graph=kws_ref_model.tflite                                                     
 Int8  | 0.321 ms        | i.MX 93      | CPU (2 threads) | benchmark_model --graph=kws_ref_model.tflite --num_threads=2                                         
 Int8  | 0.409 ms        | i.MX 93      | NPU             | benchmark_model --graph=kws_ref_model_vela.tflite --external_delegate_path=libvx_delegate.so 
 Int8  | 62 ms           | MCX N947     | CPU             | MCUXpresso SDK                                                                                                                            |
 Int8  | 3.329 ms        | MCX N947     | NPU             | MCUXpresso SDK                                                                                                                            

**Note**: Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf), to find out where benchmark_model, libvx_delegate and libethosu_delegate are located.

## Download and run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus, i.MX 93 and MCX N947 is `.tflite`.  

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

An example of how to use the model is in `example.py`.

### How to run model

```bash
python example.py --feature_typ mfcc --tfl_file_name kws_ref_model.tflite --file 'example_input.wav'
```

### Labels

Classification labels of the model:

- "Down"
- "Go"
- "Left"
- "No"
- "Off"
- "On"
- "Right"
- "Stop"
- "Up"
- "Yes"
- "Silence"
- "Unknown"

## Origin

Model implementation:

[1] Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition." arXiv preprint arXiv:
1804.03209 (2018).

[2] Banbury, Colby, et al. "Mlperf tiny benchmark." arXiv preprint arXiv:2106.07597 (2021).

[3] Zhang, Yundong, et al. "Hello edge: Keyword spotting on microcontrollers." arXiv preprint arXiv:1711.07128 (2017).
