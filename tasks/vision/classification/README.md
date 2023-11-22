# Image classification

Image classification is the task of assigning a single class (or label) to an input image.

![classification demo](./classification_demo.webp)

## Datasets

The image classification models featured in this Model Zoo are trained on the following datasets:

### ImageNet (ILSVRC)

The [ImageNet](https://www.image-net.org/) dataset [2] contains 14,197,122 annotated images according to the WordNet
hierarchy. Since 2010 the dataset is used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), a benchmark
in image classification and object detection. The publicly released dataset contains a set of manually annotated
training images. A set of test images is also released, with the manual annotations withheld.

### Facial Expression Recognition Challenge (FER2013)

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) [1] is a dataset for face expression classification.

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the
face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each
face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear,
3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples.
The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.

### CIFAR-10

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset [3] consists of 60000 color images of size 32 x 32.
Dataset contains 10 classes.

There are 50,000 training images and 10,000 test images. The dataset is divided into 5 training batches and one test
batch. Training batches contain images in random order and may contain more images from one class than others. The test
batch contain randomly selected images from every class.

## Metrics

### Categorical Accuracy

Categorical accuracy is the number of correct predictions divided by the total number of predictions

## Model List

 Model name                                       | Architecture | Backbone     | Training Dataset | Acc FP32 | Acc INT8 | Input size | OPS   | MParams | FP32 Size | INT8 Size | Compatibility                                   
--------------------------------------------------|--------------|--------------|------------------|----------|----------|------------|-------|---------|-----------|-----------|-------------------------------------------------
 [Deepface-emotion](./deepface-emotion/README.md) | CNN          | Custom       | FER2013          | 0.57     | TODO     | 48x48      | 58.5M | 1.49    | 5.8MB     | 1.5MB     | i.MX 8M Plus, i.MX 93                           
 [MobileNet V1](./mobilenetv1/README.md)          | CNN          | MobileNet V1 | ImageNet         | 0.41     | 0.39     | 128x128    | 28M   | 0.47    | 1.88MB    | 0.47MB    | i.MX 8M Plus, i.MX 93, i.MX RT1170, i.MX RT1050 
 [Tiny ResNet](./tiny-resnet/README.md)           | CNN          | ResNet       | CIFAR-10         | 0.87     | 0.75     | 32x32      | 25M   | 0.078   | 0.31MB    | 0.097MB   | MCX N947, i.MX 8M Plus, i.MX 93                 
 [Visual Wake Word](./visual-wake-word/README.md) | CNN          | MobileNet V1 | vww_coco2014     | 0.75     | 0.75     | 96x96      | 15M   | 0.221   | 0.846MB   | 0.326MB   | MCX N947, i.MX 8M Plus, i.MX 93                 
 [Mobilenet V2](./mobilenetv2/README.md)          | CNN          | MobileNet V2 | ImageNet         | 0.66     | 0.64     | 224x224    | 608M  | 3.539   | 13.7MB    | 3.9MB     | i.MX 8MP, i.MX 93                               
 [MNasNet](./mnasnet/README.md)                   | CNN          | custom       | ImageNet         | 0.77     | 0.77     | 224x224    | 447M  | 2.9M    | 11.4MB    | 3.3MB     | i.MX 8MP                                        

## References

[1] Goodfellow, Ian J., et al. "Challenges in representation learning: A report on three machine learning contests."
International conference on neural information processing. Springer, Berlin, Heidelberg, 2013.

[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and
pattern recognition. Ieee, 2009.

[3] Alex
Krizhevsky, [Learning Multiple Layers of Features from Tiny Images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf),

2009.