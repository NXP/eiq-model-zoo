# MicroSpeech model with LSTM

## Introduction

MicroSpeech model is a basic speech recognition network, that recognize two words: 'yes' and 'no'. All other words are
classified as 'unknown'. This model was designed as The TensorFlow tutorial on speech recognition networks. The tutorial
is available
on [Google Colab](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/third_party/xtensa/examples/micro_speech_lstm/train/micro_speech_with_lstm_op.ipynb)

The model was trained on the Speech Commands dataset [1]. It is an audio dataset of spoken words, for training models
that detect when a single word is spoken. It’s released under
a [Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Model Information

 Information          | Value                                                                                                                                                 
----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------
 Input shape          | Wav sound file, with shape (1, 49, 257)                                                                                                               
 Input example        | "example.wav",</br> Soundfile from Speech commands dataset [1]</br>License: [Creative Commons BY 4.0 ](https://creativecommons.org/licenses/by/4.0/). 
 Output shape         | Vector of probabilities, shape (1, 3). Labels are ['no', 'unknown', 'yes']                                                                            
 Output example       | Output tensor: [[ 127 -128 -128]                                                                                                                      
 FLOPS                | 0.023 MOPS                                                                                                                                            
 Number of parameters | 119923                                                                                                                                                
 Source framework     | Tensorflow/Keras                                                                                                                                      
 Target platform      | MPU                                                                                                                                                   

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model
(see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model is trained using `train_model.py` script. It is located in the `scripts` directory. It generates a keras
model,
which is located into the `model` directory.
The model has been trained and evaluated on the Micro speech dataset. It achieved a score of 85% accuracy on the test
set according to test, using `test_tflite_model.py` script located in the `scripts` directory.

The original training procedure is detailed in the `train_model.py` script.

## Conversion/Quantization

Quantization procedure is done by `convert_model.py` script, check
the script for more details. It generates tflite float model and quantized int8 tflite model.

## Use case and limitations

This model can be used for classification applications with very limited set of classes. Since its use-case is limited,
it should not be trusted for critical scenarios.

## Performance

Here are performance figures evaluated on i.MX 8MP using BSP LF6.1.1_1.0.0 and on i.MX 93 using BSP LF6.1.36_2.1.0:

 Model | Average latency | Platform     | Accelerator     | Command                                                                             
-------|-----------------|--------------|-----------------|-------------------------------------------------------------------------------------
 Int8  | 4.919 ms        | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=[MODEL_NAME]                                                
 Int8  | 4.915 ms        | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=[MODEL_NAME] --num_threads=4                                
 Int8  | 1.911 ms        | i.MX 8M Plus | NPU             | benchmark_model --graph=[MODEL_NAME] --external_delegate_path=libvx_delegate.so     
 Int8  | 5.203 ms        | i.MX 93      | CPU (1 threads) | benchmark_model --graph=[MODEL_NAME]                                                
 Int8  | 5.212 ms        | i.MX 93      | CPU (2 threads) | benchmark_model --graph=[MODEL_NAME] --num_threads=2                                
 Int8  | 1.461 ms        | i.MX 93      | NPU             | benchmark_model --graph=[MODEL_NAME] --external_delegate_path=libethosu_delegate.so 

**Note**: Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf), to find out
where benchmark_model, libvx_delegate and libethosu_delegate are located.

## Download and run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus and i.MX 93 is `microspeech-lstm_quant_int8.tflite`.

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If
using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download
Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version
used.

An example of how to use the model is in `example.py`.

### Labels

- 'no'
- 'unknown'
- 'yes'

## Origin

Model
implementation: [Google colab](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/third_party/xtensa/examples/micro_speech_lstm/train/micro_speech_with_lstm_op.ipynb)

[1] Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition." arXiv preprint arXiv:
1804.03209 (2018).
