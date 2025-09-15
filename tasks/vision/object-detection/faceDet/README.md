# YOLO Face Detection

## Introduction

"YOLO Face" is a lightweight face detection model designed for MCU platforms. The model is implemented based on the YOLO
algorithm, consisting of a backbone network and detection heads, and it borrows the training method from YOLOV3 [2]. The
model outputs three-dimensional vectors that can detect objects of large, medium, and small scales.

This model was developed in NXP Semiconductors.

## Model Information

| Information      | Value                                                                                                                                                                          |
|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input shape      | RGB image (160, 128, 3)                                                                                                                                                        |
| Input example    | <img src="test.jpg" > (Image source: [Public domain image](https://commons.wikimedia.org/wiki/File:Isabella_L%C3%B6vin_signing_climate_law_referral.jpg), license CC0 1.0 DEED |   
| Output shape     | (4,5,18),(8,10,18),(16,20,18)                                                                                                                                                  |
| Output example   | <img src="output.jpg" >                                                                                                                                                        |
| Parameters       | 189,654                                                                                                                                                                        |
| File size        | 287KB int8 quantized                                                                                                                                                           |
| Ram usage        | 195KB int8 quantized                                                                                                                                                           |
| Source framework | TensorFlow Lite                                                                                                                                                                |
| Target platform  | MCU, MPU                                                                                                                                                                       |

## Version and changelog

Initial release of quantized int8 TFLite model.

## Tested configurations

The int8 model has been tested on MCXN947 BRK board and i.MX RT 1060 EVK using TensorFlow Lite Micro.

## Evaluation

The model hase been trained and evaluated on WiderFace dataset. It achieved scores of mAP 0.53 on the easy sets.

## Use case and limitations

The model can be used for fast face detection on low-cost MCU platforms.

## Run

### Requirements

For requirements see `requirements.txt` file or install with pip:

```
pip install -r requirements.txt
```

### How to Run

The TFLite model file for i.MX 8M Plus and for i.MX 93 is `yolo_face_detect.tflite`.

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

example.py support detection from a static image or a camera of users laptop.

```
python example.py -image xxx.jpg 
(default is test.jpg)
```

Input image:

![input](test.jpg)

Output image:

![output](output.jpg)

or

```
python example.py -video
```

## Reference

https://github.com/qqwweee/keras-yolo3

[1] WIDERface dataset: Yang, Shuo, et al. "Wider face: A face detection benchmark." Proceedings of the IEEE conference
on
computer vision and pattern recognition. 2016.

[2] Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
