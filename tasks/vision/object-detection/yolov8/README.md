# YOLOv8-m

## Introduction

Yolov8 is a state-of-the-art computer vision model, built upon previous Yolo models. The model can be used for object
detection, classification and segmentation tasks. This repository contains model for object detection task.

Yolov8 is built on the Yolov5 model. Compared to Yolov5, Yolov8 comes with a new anchor-free detection system, changes
to the convolutional blocks in the model and mosaic augmentation during training. [2]

This model was developed by Ultralytics and is released under AGPL-3.0 license for open source purposes. In case, one
would like to use the model for commercial purpose, refer
to [Ultralytics Enterprise license](https://www.ultralytics.com/license).

The model is released under different sizes: Yolov8-n, Yolov8-s. Yolov8-m, Yolov8-l and Yolov8-x. This repository contains information about Yolov8-m.


## Model Information

 Information          | Value                                                                                                                                            
----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------
 Input shape          | Color image (320, 320, 3)                                                                                                                        
 Input example        | <img src="example_input.jpg" width=320px> ([Image source](https://commons.wikimedia.org/wiki/File:Moscow_bus_151872_2022-05.jpg), Public domain) 
 Output shape         | Tensor of size (1,84,8400)                                            
 Output example       | <img src="example_output.jpg" width=320px>                                                                                                       
 FLOPS                | 78.9 B OPS                                                                                                                                         
 Number of parameters | 25.9M                                                                                                                                             
 Source framework     | Pytorch                                                                                                                                          
 Target platform      | MPU, MCU                                                                                                                                         

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the COCO dataset. It achieved a score of 50.2 mAP on the test set according
to [the documentation](https://docs.ultralytics.com/datasets/detect/coco/).

The original training procedure is
detailed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py).

## Conversion/Quantization

The model is downloaded in the PyTorch format through Ultralytics module. Using Ultralytics command line application,
model is converted and fully quantized into tflite format. For more details, please refer to
the [source code of the model](https://github.com/ultralytics/ultralytics/tree/main).

## Use case and limitations

This model can be used for object detection applications.

## Download and run

### How to get model

0. Check license options at Ultralytics page - AGPL 3.0 or commercial
1. install package with following command:

```bash
pip install ultralytics
```

2. get int8 quantized tflite model with following command:

```bash
yolo export model=yolov8m.pt imgsz=320 format=tflite int8
```

3. The TFLite model file for i.MX 8M Plus and for i.MX 93 is `yolov8m_full_integer_quant.tflite` located in
   the `yolov8m_saved_model` directory.

#### Inference

Use following command to run inference:

````bash
yolo detect predict model=./yolov8m_saved_model/yolov8m_full_integer_quant.tflite source='https://ultralytics.com/images/bus.jpg'
````

Result is saved into /runs/detect/predic directory.

To see example inference script, visit [Ultralytics Example github pages](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-int8-tflite-Python/main.py)

## Origin

[1] Original model implementation: https://github.com/ultralytics/ultralytics

[2] Model documentation: https://docs.ultralytics.com/

[4] Dataset: https://cocodataset.org/#home

[5] Jocher, Glenn, and Chaurasia, Ayush and Jing Qiu. Ultralytics YOLOv8. 8.0.0,
2023, https://github.com/ultralytics/ultralytics
