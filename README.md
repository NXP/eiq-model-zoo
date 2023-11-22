# NXP eIQ® Model Zoo

> A collection of machine learning models for computer vision applications, audio applications and others, optimized for NXP MCUs and MPUs.

## Description

The NXP eIQ® Model Zoo offers pre-trained models for a variety of domains and tasks that are ready to be deployed on supported products.

Models are included in the form of "recipes" that convert the original models to TensorFlow Lite format.
This allows users to find the original re-trainable versions of the models, allowing fine-tuning/training if required.

This should facilitate the development of embedded AI with NXP products.

## Supported Products and Machine Learning Frameworks

Models are provided in TensorFlow Lite format.

The list of currently supported platforms can be found in [the products folder](./products/).

## NXP eIQ® Model Zoo layout

The NXP eIQ® Model Zoo is structured in the following way: Main Page -> Domain -> Task -> Model.

Multiple models may be proposed for the same task. Each model has its own information page.

### List of domains

#### [Audio](./tasks/audio/README.md)

- [anomaly detection](./tasks/audio/anomaly-detection/README.md)
- [command recognition](./tasks/audio/command-recognition/README.md)

#### [Vision](./tasks/vision/README.md)

- [classification](./tasks/vision/classification/README.md)
- [face recognition](./tasks/vision/face-recognition/README.md)
- [object detection](./tasks/vision/object-detection/README.md)
- [pose estimation](./tasks/vision/pose-estimation/README.md)
- [semantic segmentation](./tasks/vision/semantic-segmentation/README.md)

#### [Misc. - other domains](./tasks/misc/README.md)



## Requirements

The model creation recipes were tested under Ubuntu 20.04 with Python 3.8.

## Legal information

Copyright NXP 2023

The code in this repository is provided under [MIT License](https://choosealicense.com/licenses/mit/).
