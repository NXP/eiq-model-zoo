# Instance Segmentation

Instance Segmentation is the task of detecting instances of objects of a certain class within an image and creating a pixel-wise mask of each object. A bounding box, a class label and a mask are found for each detected object.

## Datasets

The Instance Segmentation models featured in this Model Zoo are trained on the following datasets.

### COCO

MS COCO[1] is a large-scale Instance Segmentation, segmentation, and captioning dataset.

The COCO train, validation, and test sets, containing more than 200,000 images and 80 object categories, are available on the download page. All object instances are annotated with a detailed segmentation mask. Annotations on the training
and validation sets (with over 500,000 object instances segmented) are publicly available.

## Metrics

### mAP

Average Precision (AP) is defined as an area under
the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall) curve. For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match).
Here, it is specifically the mean of the mask AP and the bounding box AP.

## Model List

 Model name                                             | Architecture | Backbone             | Training Dataset | mAP FP32                            | mAP INT8          | Input size | OPS  | Params | FP32 Size | INT8 Size | Compatibility                            
--------------------------------------------------------|--------------|----------------------|------------------|-------------------------------------|-------------------|------------|------|--------|-----------|-----------|------------------------------------------
 [YOLACT-Edge](./YOLACT-Edge/README.md)           | YOLACT[2]       | Mobilenetv2[3] | COCO       | 0.21	 | TODO              | 550x550    | 17G |  9M  | 32MB    | 8.5MB     | i.MX 8M Plus, i.MX 93        

## References

[1] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer,
Cham, 2014.

[2] Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee. Yolact: real-time instance segmentation. In ICCV, 2019.

[3] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference
on computer vision and pattern recognition. 2018.