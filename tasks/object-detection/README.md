# Object Detection

 Object detection is the task of detecting instances of objects of a certain class within an image. A bounding box and a class label are found for each detected object.

 ![detection demo](./detection_demo.webp)

## Datasets

The object detection models featured in this Model Zoo are trained on the following datasets.

### COCO

MS COCO[1] is a large-scale object detection, segmentation, and captioning dataset.

The COCO Object Detection Task is designed to push the state of the art in object detection forward. COCO features two object detection tasks: using either bounding box output or object segmentation output (the latter is also known as instance segmentation).

The COCO train, validation, and test sets, containing more than 200,000 images and 80 object categories, are available on the download page. All object instances are annotated with a detailed segmentation mask. Annotations on the training and validation sets (with over 500,000 object instances segmented) are publicly available.

### WIDER FACE

WIDER FACE[2] dataset is a face detection benchmark dataset, of which images are selected from the publicly available WIDER dataset. We choose 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. WIDER FACE dataset is organized based on 61 event classes.

## Metrics

### mAP

Average Precision (AP) is defined as an area under the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall) curve.

## Model List

Model name                                   | Architecture | Backbone              | Training Dataset | mAP FP32 | mAP INT8 | Input size | OPS    | Params    | FP32 Size    | INT8 Size |  Compatibility
---                                          | ---          |     ---               | ---              | ---      | ---      | ---        |  ---    | ---        |  ---         |    ---    | ---
[UltraFace-slim](./ultraface-slim/README.md) | SSD[3]       | Custom SqueezeNet[5]  | WIDER FACE       | 0.77/0.671/0.395 (easy/medium/hard)   | TODO     | 320x240    |  168M | 265K    |  1.04MB      |  300KB    | i.MX 8M Plus, i.MX 93, RT1170
[NanoDet-M](./nanodet-m/README.md)           | FCOS[6]      | ShuffleNetV2[7] 0.5x  | COCO[1]          | <0.13    | 0.04     | 320x320    | 158M   | 204K      |     1.6MB    |  364KB    | i.MX 8MP, RT1170
[YOLOv4-tiny](./yolov4tiny/README.md)        | YOLOv4[8]    | CSPDarkNet53[8]       | COCO[1]          | 0.40 (mAP@0.5IoU) | 0.33 (mAP@0.5IoU)    | 416x416    | 6.9G   | 6.05M     |     24MB     |  5.9MB    | i.MX 8M Plus, i.MX 93
[SSDLite MobileNetV2](./ssdlite-mobilenetv2/README.md) | SSD[3] | MobileNetV2 [4]   | COCO[1]          | 0.22     | 0.16 (val)     | 300x300    | 1.5G   | 4.3M      | 20MB         |   5.4MB   | i.MX 8M Plus, i.MX 93

## References

[1] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.

[2] Yang, Shuo, et al. "Wider face: A face detection benchmark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[3] Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.

[4] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[5] Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size." arXiv preprint arXiv:1602.07360 (2016).

[6] Tian, Zhi, et al. "Fcos: Fully convolutional one-stage object detection." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

[7] Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the European conference on computer vision (ECCV). 2018.

[8] Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).