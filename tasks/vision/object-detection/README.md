# Object Detection

Object detection is the task of detecting instances of objects of a certain class within an image. A bounding box and a
class label are found for each detected object.

![detection demo](./detection_demo.webp)

## Datasets

The object detection models featured in this Model Zoo are trained on the following datasets.

### COCO

MS COCO[1] is a large-scale object detection, segmentation, and captioning dataset.

The COCO Object Detection Task is designed to push the state of the art in object detection forward. COCO features two
object detection tasks: using either bounding box output or object segmentation output (the latter is also known as
instance segmentation).

The COCO train, validation, and test sets, containing more than 200,000 images and 80 object categories, are available
on the download page. All object instances are annotated with a detailed segmentation mask. Annotations on the training
and validation sets (with over 500,000 object instances segmented) are publicly available.

### WIDER FACE

WIDER FACE[2] dataset is a face detection benchmark dataset, of which images are selected from the publicly available
WIDER dataset. We choose 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and
occlusion as depicted in the sample images. WIDER FACE dataset is organized based on 61 event classes.

### PASCAL VOC 2012
The PASCAL Visual Object Classes (VOC) 2012 dataset [10] is a large-scale object detection and semantic segmentation dataset.

The PASCAL Visual Object Classes (VOC) 2012 dataset [10] is a large-scale object detection and semantic segmentation
dataset.

It contains contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat,
bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse,
sheep, and person. Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and
object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation,
and classification tasks. The PASCAL VOC dataset is split into three subsets: 1,464 images for training, 1,449 images
for validation and a private testing set.

## Metrics

### mAP

Average Precision (AP) is defined as an area under
the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall) curve.

## Model List

 Model name                                             | Architecture     | Backbone             | Training Dataset | mAP FP32                            | mAP INT8                   | Input size | OPS     | Params | FP32 Size | INT8 Size | Compatibility                            
--------------------------------------------------------|------------------|----------------------|------------------|-------------------------------------|----------------------------|------------|---------|--------|-----------|-----------|------------------------------------------
 [UltraFace-slim](./ultraface-slim/README.md)           | SSD[3]           | Custom SqueezeNet[5] | WIDER FACE       | 0.77/0.671/0.395 (easy/medium/hard) | TODO                       | 320x240    | 168M    | 265K   | 1.04MB    | 300KB     | i.MX 8M Plus, i.MX 93, RT1170            
 [UltraFace-ultraslim](./ultraface-ultraslim/README.md) | SSD[3]           | Custom SqueezeNet[5] | WIDER FACE       | 0.77/0.671/0.395 (easy/medium/hard) | TODO                       | 128x128    | 34.5M   | 262K   | N/A       | 375KB     | MCXN947                                  
 [NanoDet-M](./nanodet-m/README.md)                     | FCOS[6]          | ShuffleNetV2[7] 0.5x | COCO[1]          | <0.13                               | 0.04                       | 320x320    | 158M    | 204K   | 1.6MB     | 364KB     | i.MX 8MP, RT1170                         
 [YOLOv4-tiny](./yolov4tiny/README.md)                  | YOLOv4[8]        | CSPDarkNet53[8]      | COCO[1]          | 0.40 (mAP@0.5IoU)                   | 0.33 (mAP@0.5IoU)          | 416x416    | 6.9G    | 6.05M  | 24MB      | 5.9MB     | i.MX 8M Plus, i.MX 93                    
 [SSDLite MobileNetV2](./ssdlite-mobilenetv2/README.md) | SSD[3]           | MobileNetV2 [4]      | COCO[1]          | 0.22                                | 0.16 (val)                 | 300x300    | 1.5G    | 4.3M   | 20MB      | 5.4MB     | i.MX 8M Plus, i.MX 93                    
 [FaceDet](./faceDet/README.md)                         | YoloV3[9]        | CSPDarkNet53[8]      | WIDER FACE       | -                                   | 0.53                       | 160x128    | -       | 189K   | -         | 287KB     | i.MX 8M Plus, i.MX 93, RT1060,   MCXN947 
 [FastestDet](./fastestDet/README.md)                   | FastestDet       | ShuffleNetV2[7] 1.0x | PASCAL VOC [10]  | 0.5742 (val) (AP@0.5 IoU)           | 0.5736 (val) (mAP@0.5 IoU) | 220x220    | 65.593M | 152.9K | 659 KB    | 315 KB    | RT1170                                   
 [CenterNet](./centernet/README.md)                     | CenterNet[11]    | MobileNetV2 [4]      | COCO[1]          | 0.20                                | 0.17                       | 512x512    | 4.7G    | 2.36M  | 9.0MB     | 2.8MB     | i.MX 8M Plus, i.MX 93                    
 [Yolov5](./yolov5/README.md)                           | Yolov5[13]       | CSPDarkNet53[8]      | COCO[1]          | 0.771 (mAP@0.5IoU)                  | 0.737 (mAP@0.5IoU)         | 640x640    | 49.0G   | 21.2M  | 40.5 MB    | 20.8 MB   | i.MX 8M Plus, i.MX 93                    
 [EfficientDet-lite0](./efficientdet-lite0/README.md)   | EfficientDet[14] | EfficientDet[14]     | COCO[1]          | 26.41                               | 26.10                      | 320x320    | 1.9G    | 3.2M   | -         | 4.34MB    | i.MX 93, i.MX 8M Plus                    
 [Yolov8](./yolov8/README.md)                           | Yolov8[12]   | custom               | COCO[1]          | 0.502 (mAP@0.5-0.95IoU)             | 0.448 (mAP@0.5IoU) | 320x320    | 78.9B   | 25.9M  | 101.4 MB  | 25.5 MB   | i.MX 8M Plus, i.MX 93

## References

[1] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer,
Cham, 2014.

[2] Yang, Shuo, et al. "Wider face: A face detection benchmark." Proceedings of the IEEE conference on computer vision
and pattern recognition. 2016.

[3] Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.

[4] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference
on computer vision and pattern recognition. 2018.

[5] Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size."
arXiv preprint arXiv:1602.07360 (2016).

[6] Tian, Zhi, et al. "Fcos: Fully convolutional one-stage object detection." Proceedings of the IEEE/CVF international
conference on computer vision. 2019.

[7] Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the
European conference on computer vision (ECCV). 2018.

[8] Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object
detection." arXiv preprint arXiv:2004.10934 (2020).

[9] Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).

[10] Everingham, Mark, et al. "The pascal visual object classes (voc) challenge." International journal of computer vision 88 (2010): 303-338.

[11] Xingyi Zhou, Dequan Wang, and Philipp Krahenbuhl. Objects as points. arXiv preprint arXiv:1904.07850, 2019.

[12] Jocher, Glenn, and Chaurasia, Ayush and Jing Qiu.  Ultralytics YOLOv8. 8.0.0, 2023, https://github.com/ultralytics/ultralytics
