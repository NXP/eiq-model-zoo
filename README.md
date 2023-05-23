# NXP eIQ® Vision Model Zoo

> A collection of machine learning models for computer vision applications, optimized for NXP MCUs and MPUs.

## Description

The NXP eIQ® Vision Model Zoo offers pre-trained models for a variety of domains and tasks that are ready to be deployed on supported products.

Models are included in the form of "recipes" that convert the original models to TensorFlow Lite format.
This allows users to find the original re-trainable versions of the models, allowing fine-tuning/training if required.

This should facilitate the development of embedded AI with NXP products.

## Supported Products and Machine Learning Frameworks

Models are provided in TensorFlow Lite format.

The list of currently supported platforms can be found in [the products folder](./products/).

## NXP eIQ® Vision Model Zoo layout

The NXP eIQ® Vision Model Zoo is structured in the following way: Main Page -> Task -> Model.

Multiple models may be proposed for the same task. Each model has its own information page.

## List of tasks

The Vision domain includes all tasks related to image and video processing. Below is a table of the currently supported tasks.
Follow the links to each domain to access the models.

Task | Description | Input Type | Output Type | Example
---  | ---         | ---        | ---         | ---
[Image Classification](./tasks/classification/) | Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. <br> The goal is to classify the image by assigning it to a specific label. <br> Typically, Image Classification refers to images in which only one object appears and is analyzed. | Image | Label |  <img src="./tasks/classification/classification_demo.webp"  width="200">
[Object Detection](./tasks/object-detection/) | Object detection is the task of detecting instances of objects of a certain class within an image. <br> A bounding box and a class label are found for each detected object.  | Image | Bounding Boxes + Labels | <img src="./tasks/object-detection/detection_demo.webp"  width="200">
[Face Recognition](./tasks/face-recognition/) | Face recognition is the task of matching an input face image to a databases of known faces. <br> A face feature vector is regressed by the model and compared to the known feature vectors. | Image | Face feature vector | <img src="./tasks/face-recognition/face_demo.webp"  width="200">

## Legal information

Copyright NXP 2023

The code in this repository is provided under [MIT License](https://choosealicense.com/licenses/mit/).
