# Monocular Depth Estimation

The goal of monocular depth estimation is to estimate the depth of the scene at each pixel of a single input image. It is very useful in applications such as robotics.

![pose estimation](./midas/example_input.jpg) ![pose estimation](./midas/example_output.jpg)

## Datasets

The monocular depth estimation models featured in this Model Zoo are trained on the following datasets: RedWeb, MegaDepth, WSVD, 3D Movies, DIML indoor, HRWSI, IRS, TartanAir, BlendedMVS, ApolloScape.
Definitions of these datasets can be found in the [MiDaS repository](https://github.com/isl-org/MiDaS).

## Metrics

Various metrics are used by the authors of MiDaS depending on the type of ground truth used in each dataset. We refer the user to page 7 of the original paper [1] for more details. 

## Model List

Model name                                | Architecture            | Backbone              | Training Dataset | Accuracy (see [original repo](https://github.com/isl-org/MiDaS))            | Input size | OPS    | Params  | INT8 Size |  Compatibility
---                                       | ---                     |     ---               | ---              | ---                 | ---        |  ---   | ---     |    ---    | ---
[MiDaS v2.1 Small](./midas/README.md)     | MiDaS 2.1 [1]           | EfficientNet-Lite3[2] | 10 datasets      | -76                 | 256x256    | 9.214G | 21M     |  17M      | i.MX 8M Plus, i.MX 93

## References

[1] Ranftl, René, et al. "Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer." IEEE transactions on pattern analysis and machine intelligence 44.3 (2020): 1623-1637.

[2] https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html

[3] Ranftl, René, Alexey Bochkovskiy, and Vladlen Koltun. "Vision transformers for dense prediction." Proceedings of the IEEE/CVF international conference on computer vision. 2021.