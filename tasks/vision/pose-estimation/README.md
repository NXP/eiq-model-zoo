# Pose Estimation

The goal of pose estimation is to detect the position and orientation of a person, face, or object. In Human Pose Estimation, this is usually done with specific keypoints such as hands, head, legs, etc.

 ![pose estimation](./pose_demo.webp)

## Datasets

The pose estimation models featured in this Model Zoo are trained on the following datasets:

### COCO Keypoints

The MS COCO (Microsoft Common Objects in Context) dataset [1] is a large-scale object detection, segmentation, key-point detection, and captioning dataset.

The dataset contains more than 200,000 images and 250,000 person instances labeled with keypoints (17 possible keypoints, such as left eye, nose, right hip, right ankle).

### Active dataset

Active is a proprietary dataset from Google, featuring images sampled from YouTube videos of people exercising. It contains 23.5k images.

### 300W-LP

300W (300 Faces-In-The-Wild) [5] is a face dataset that consists of 300 Indoor and 300 Outdoor in-the-wild images. 
300W-LP [4] is expanded from 300W, which standardises multiple alignment databases with 68 landmarks, including AFW, LFPW, HELEN, IBUG and XM2VTS. 
With 300W, 300W-LP adopt the proposed face profiling to generate 61,225 samples across large poses (1,786 from IBUG, 5,207 from AFW, 16,556 from LFPW and 37,676 from HELEN, XM2VTS is not used).

The dataset can be employed as the training set for the following computer vision tasks: face attribute recognition and landmark (or facial part) localization.
Thus, it can also be used for head pose estimation training.

[Dataset description source](https://www.tensorflow.org/datasets/catalog/the300w_lp).

### Internal Intel face landmark dataset

As described in the [OpenVino Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/facial-landmarks-35-adas-0002), this dataset is comprised of 1000 images from 300 different people with various face expressions.
It is not publicly available.

## Metrics

### Keypoint mAP

The COCO keypoint mAP mimics the detection mAP and is based on the Object Keypoint Similarity metric. More details can be found [here](https://cocodataset.org/#keypoints-eval).

### MAE (Mean Absolute Error)

The MAE is used to evaluate the average prediction error in regression tasks. In the case of head pose estimation, it is the average error of yaw, pitch and roll.

## Model List

Model name                                                                 | Architecture | Backbone       | Training Dataset | Accuracy            | Input size | OPS    | Params  | INT8 Size |  Compatibility
---                                                                        | ---          |     ---        | ---              | ---                 | ---        |  ---   | ---     |    ---    | ---
[MoveNet](./movenet/README.md)                                             | MoveNet      | MobileNetV2[3] | COCO[1] + Active | 57.4 (mAP int8)     | 192x192    |  N/A   | N/A     |  2.9M     | i.MX 8M Plus, i.MX 93
[WHENet](./whenet/README.md)                                               | WHENet       | EfficientNetB0 | 300W-LP [4]      | 4.619 (MAE float32) | 224x224    | 781M   | 4.4M    |  5.0M     | i.MX 8M Plus, i.MX 93
[facial-landmarks-35-adas-0002](./facial-landmarks-35-adas-0002/README.md) | Custom CNN   | Custom         | Private          | 0.106 (MNE float32) | 60x60      | 42M    | 4.595M  |  4.7M     | i.MX 8M Plus, i.MX 93

## References

[1] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.

[2] https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html

[3] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[4] Zhu, Xiangyu, et al. "Face alignment across large poses: A 3d solution." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[5] Sagonas, Christos, et al. "300 faces in-the-wild challenge: The first facial landmark localization challenge." Proceedings of the IEEE international conference on computer vision workshops. 2013.