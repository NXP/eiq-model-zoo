# Face recognition

Face recognition (also called face identification) is the task of matching an input face with the closest face present in a database of faces.

This is done by regressing a face feature vector for the input image and comparing it to the feature vectors of the database, using metrics such as euclidean distance or cosine similarity.

Models created for face recognition can also be used for some simple form of face verification (i.e. checking that two faces are the same). However, they should **not** be trusted for security-related applications.

## Datasets

The object detection models featured in this Model Zoo are trained on the following datasets.

### LFW (Labeled Faces in the Wild)

LFW [1] is a database of face photographs designed for studying the problem of unconstrained face recognition. The data set contains more than 13,000 images of faces collected from the web. Each face has been labeled with the name of the person pictured. 1680 of the people pictured have two or more distinct photos in the data set.

## Metrics

### Accuracy

Accuracy is equal to (TP + TN) / (FP + FN + TP + TN), where T=True, F=False, P=Positive, N=Negative.

### AUC

AUC is defined as an area under the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve.

## Model List

Model name         | Architecture | Backbone              | Training Dataset | Acc FP32 | Acc INT8 | Input size | OPS    | MParams    | FP32 Size    | INT8 Size |  Compatibility
---                | ---          |     ---               | ---              | ---      | ---      | ---        |  ---    | ---        |  ---         |    ---    | ---
[FaceNet512](./facenet512/README.md)[3] | FaceNet[3]      | Inception-Resnet-V2[2]  | LFW[1]       | 0.975 (LFWpairsTest)   | 0.972 (LFWpairsTest)  | 160x160   |  2.84G   | 23M   |  91M      |  24M    | i.MX 8M Plus, i.MX 93

## References

[1] Huang, Gary B., et al. "Labeled faces in the wild: A database forstudying face recognition in unconstrained environments." Workshop on faces in'Real-Life'Images: detection, alignment, and recognition. 2008.

[2] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." Thirty-first AAAI conference on artificial intelligence. 2017.

[3] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.