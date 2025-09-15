# NXP eIQ® Model Zoo supported platforms

> A collection of machine learning models for computer vision applications, audio applications and others, optimized for
> NXP MCUs and MPUs.

This page contains a list of each supported platform with its supported models.

## i.MX 8M Plus

| Model                                                                                                    | Task                    | Dataset         |
|----------------------------------------------------------------------------------------------------------|-------------------------|-----------------|
| [Deepface-emotion](../tasks/vision/classification/deepface-emotion/README.md)                            | Image classification    | FER2013         |
| [Facenet512](../tasks/vision/face-recognition/facenet512/README.md)                                      | Face recognition        | LFW             |
| [NanoDet-M](../tasks/vision/object-detection/nanodet-m/README.md)                                        | Object detection        | COCO            |
| [YOLOv4-tiny](../tasks/vision/object-detection/yolov4tiny/README.md)                                     | Object detection        | COCO            |
| [Ultraface-Slim](../tasks/vision/object-detection/ultraface-slim/README.md)                              | Object detection        | Widerface       |
| [SSDLite MobileNetV2](../tasks/vision/object-detection/ssdlite-mobilenetv2/README.md)                    | Object detection        | COCO            |
| [MobileNet V1](../tasks/vision/classification/mobilenetv1/README.md)                                     | Image classification    | ImageNet        |
| [DeepLabV3](../tasks/vision/semantic-segmentation/deeplabv3/README.md)                                   | Semantic segmentation   | PASCAL VOC 2012 |
| [MoveNet](../tasks/vision/pose-estimation/movenet/README.md)                                             | Pose estimation         | COCO + Active   |
| [Selfie-Segmenter](../tasks/vision/segmentation/selfie-segmenter/README.md)                              | Semantic Segmentation   | Proprietary     |
| [WHENet](../tasks/vision/pose-estimation/whenet/README.md)                                               | Pose estimation (head)  | 300W-LP         |
| [facial-landmarks-35-adas-0002](../tasks/vision/pose-estimation/facial-landmarks-35-adas-0002/README.md) | Pose estimation (face)  | Proprietary     |
| [Tiny-ResNet](../tasks/vision/classification/tiny-resnet/README.md)                                      | Image classification    | CIFAR-10        |
| [MobileNet V2](../tasks/vision/classification/mobilenetv2/README.md)                                     | Image classification    | Imagenet        |
| [MnasNet](../tasks/vision/classification/mnasnet/README.md)                                              | Image classification    | Imagenet        |
| [EfficientNet-lite](../tasks/vision/classification/efficientnet-lite/README.md)                          | Image classification    | Imagenet        |
| [Visual Wake Word](../tasks/vision/classification/visual-wake-word/README.md)                            | Image classification    | VW COCO2014     |
| [EegTCNet](../tasks/misc/eegTCNet/README.md)                                                             | Others / classification | 4 motor imagery |
| [MicroSpeech LSTM](../tasks/audio/command-recognition/micro-speech-LSTM/README.md)                       | Command recognition     | Speech commands |
| [Keyword spotting](../tasks/audio/commands-recognition/keyword-spotting_DSCNN/README.md)                 | Command recognition     | Speech commands |
| [Anomaly detection AD](../tasks/audio/anomaly-detection/deep-autoencoder/README.md)                      | Anomaly detection       | ToyADMOS        |
| [FaceDet](../tasks/vision/object-detection/faceDet/README.md)                                            | Object detection        | WiderFace       |
| [MiDaS v2.1 Small](../tasks/vision/monocular-depth-estimation/midas/README.md)                           | Mono Depth Estimation   | 10 datasets     |
| [wav2letter](../tasks/audio/speech-recognition/wav2letter/README.md)                                     | Speech recognition      | LibriSpeech     |
| [ResNet50](../tasks/vision/classification/resnet/README.md)                                              | Image classification    | Imagenet        |
| [Fast-SRGAN](../tasks/vision/super-resolution/Fast-SRGAN/README.md)                                      | Super resolution        | DIV2k           |
| [SCI](../tasks/vision/low-light-enhancement/SCI/README.md)                                               | Low-light enhancement   | LOL             |
| [YOLACT-Edge](../tasks/vision/instance-segmentation/YOLACT-Edge/README.md)                               | Instance segmentation   | COCO            |
| [CenterNet](../tasks/vision/object-detection/centernet/README.md)                                        | Object detection        | COCO            |
| [Inception v4](../tasks/vision/classification/inceptionv4/README.md)                                     | Image classification    | Imagenet        |
| [Yolov8](../tasks/vision/object-detection/yolov8/README.md)                                              | Object detection        | COCO            |
| [Yolov5](../tasks/vision/object-detection/yolov5/README.md)                                              | Object detection        | COCO            |
| [EfficientDet-lite0](../tasks/vision/object-detection/efficientdet-lite0/README.md)                      | Object detection        | COCO            |

## i.MX 93

| Model                                                                                                    | Task                    | Dataset         |
|----------------------------------------------------------------------------------------------------------|-------------------------|-----------------|
| [Deepface-emotion](../tasks/vision/classification/deepface-emotion/README.md)                            | Image classification    | FER2013         |
| [Facenet512](../tasks/vision/face-recognition/facenet512/README.md)                                      | Face recognition        | LFW             |
| [Ultraface-Slim](../tasks/vision/object-detection/ultraface-slim/README.md)                              | Object detection        | Widerface       |
| [SSDLite MobileNetV2](../tasks/vision/object-detection/ssdlite-mobilenetv2/README.md)                    | Object detection        | COCO            |
| [MobileNet V1](../tasks/vision/classification/mobilenetv1/README.md)                                     | Image classification    | ImageNet        |
| [MoveNet](../tasks/vision/pose-estimation/movenet/README.md)                                             | Pose estimation         | COCO + Active   |
| [YOLOv4-tiny](../tasks/vision/object-detection/yolov4tiny/README.md)                                     | Object detection        | COCO            |
| [Selfie-Segmenter](../tasks/vision/segmentation/selfie-segmenter/README.md)                              | Semantic Segmentation   | Proprietary     |
| [WHENet](../tasks/vision/pose-estimation/whenet/README.md)                                               | Pose estimation (head)  | 300W-LP         |
| [facial-landmarks-35-adas-0002](../tasks/vision/pose-estimation/facial-landmarks-35-adas-0002/README.md) | Pose estimation (face)  | Proprietary     |
| [Tiny-ResNet](../tasks/vision/classification/tiny-resnet/README.md)                                      | Image classification    | CIFAR-10        |
| [MobileNet V2](../tasks/vision/classification/mobilenetv2/README.md)                                     | Image classification    | Imagenet        |
| [EfficientNet-lite](../tasks/vision/classification/efficientnet-lite/README.md)                          | Image classification    | Imagenet        |
| [Visual Wake Word](../tasks/vision/classification/visual-wake-word/README.md)                            | Image classification    | VW COCO2014     |
| [EegTCNet](../tasks/misc/eegTCNet/README.md)                                                             | Others / classification | 4 motor imagery |
| [MicroSpeech LSTM](../tasks/audio/command-recognition/micro-speech-LSTM/README.md)                       | Command recognition     | Speech commands |
| [Keyword spotting](../tasks/audio/commands-recognition/keyword-spotting_DSCNN/README.md)                 | Command recognition     | Speech commands |
| [Anomaly detection AD](../tasks/audio/anomaly-detection/deep-autoencoder/README.md)                      | Anomaly detection       | ToyADMOS        |
| [FaceDet](../tasks/vision/object-detection/faceDet/README.md)                                            | Object detection        | WiderFace       |
| [MiDaS v2.1 Small](../tasks/vision/monocular-depth-estimation/midas/README.md)                           | Mono Depth Estimation   | 10 datasets     |
| [wav2letter](../tasks/audio/speech-recognition/wav2letter/README.md)                                     | Speech recognition      | LibriSpeech     |
| [ResNet50](../tasks/vision/classification/resnet/README.md)                                              | Image classification    | Imagenet        |
| [Fast-SRGAN](../tasks/vision/super-resolution/Fast-SRGAN/README.md)                                      | Super resolution        | DIV2k           |
| [YOLACT-Edge](../tasks/vision/instance-segmentation/YOLACT-Edge/README.md)                               | Instance segmentation   | COCO            |
| [SCI](../tasks/vision/low-light-enhancement/SCI/README.md)                                               | Low-light enhancement   | LOL             |
| [CenterNet](../tasks/vision/object-detection/centernet/README.md)                                        | Object detection        | COCO            |
| [Inception v4](../tasks/vision/classification/inceptionv4/README.md)                                     | Image classification    | Imagenet        |
| [Yolov8](../tasks/vision/object-detection/yolov8/README.md)                                              | Object detection        | COCO            |
| [Yolov5](../tasks/vision/object-detection/yolov5/README.md)                                              | Object detection        | COCO            |
| [EfficientDet-lite0](../tasks/vision/object-detection/efficientdet-lite0/README.md)                      | Object detection        | COCO            |

## i.MX RT1170

| Model                                                                       | Task                 | Dataset   |
|-----------------------------------------------------------------------------|----------------------|-----------|
| [NanoDet-M](../tasks/vision/object-detection/nanodet-m/README.md)           | Object detection     | COCO      |
| [Ultraface-Slim](../tasks/vision/object-detection/ultraface-slim/README.md) | Object detection     | Widerface |
| [MobileNet V1](../tasks/vision/classification/mobilenetv1/README.md)        | Image classification | ImageNet  |
| [FastestDet](../tasks/vision/object-detection/fastestDet/README.md)         | Object detection     | PASCAL VOC|

## i.MX RT1050

| Model                                                                       | Task                 | Dataset   |
|-----------------------------------------------------------------------------|----------------------|-----------|
| [NanoDet-M](../tasks/vision/object-detection/nanodet-m/README.md)           | Object detection     | COCO      |
| [Ultraface-Slim](../tasks/vision/object-detection/ultraface-slim/README.md) | Object detection     | Widerface |
| [MobileNet V1](../tasks/vision/classification/mobilenetv1/README.md)        | Image classification | ImageNet  |

## i.MX RT1060

| Model                                                         | Task             | Dataset   |
|---------------------------------------------------------------|------------------|-----------|
| [FaceDet](../tasks/vision/object-detection/faceDet/README.md) | Object detection | WiderFace |

## MCX N947

| Model                                                                                    | Task                 | Dataset         |
|------------------------------------------------------------------------------------------|----------------------|-----------------|
| [Tiny-ResNet](../tasks/vision/classification/tiny-resnet/README.md)                      | Image classification | CIFAR-10        |
| [Visual Wake Word](../tasks/vision/classification/visual-wake-word/README.md)            | Image classification | VW COCO2014     |
| [Keyword spotting](../tasks/audio/commands-recognition/keyword-spotting_DSCNN/README.md) | Command recognition  | Speech commands |
| [Anomaly detection AD](../tasks/audio/anomaly-detection/deep-autoencoder/README.md)      | Anomaly detection    | ToyADMOS        |
| [FaceDet](../tasks/vision/object-detection/faceDet/README.md)                            | Object detection     | WiderFace       |
| [Ultraface-Ultraslim](../tasks/vision/object-detection/ultraface-ultraslim/README.md)    | Object detection     | WiderFace       |