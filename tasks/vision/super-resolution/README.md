# Super-Resolution

From Paperswithcode: "Super-Resolution is a task in computer vision that involves increasing the resolution of an image or video by generating missing high-frequency details from low-resolution input. The goal is to produce an output image with a higher resolution than the input image, while preserving the original content and structure.
( Credit: MemNet )"

## Datasets

The Super-Resolution models featured in this Model Zoo are trained on the following datasets.

### DIV2k

DIV2k[1] is a large-scale Super-Resolution dataset.

It is comprised of 1000 High Resolution images and their downscaled variant. The Dataset contains multiple folders for each each downscaling factors it supports (x2, x3, x4) and for each degradation operator (bicubic and unknown for blind Super-Resolution). It is adequate for the two main tasks of Super-Resolution: blind Super-Resolution (where the model tries to guess the overarching structure of the operator used to downscale its training images) and "standard" Super-Resolution where the operator is already assumed.

## Metrics

### PSNR

Peak signal-to-noise ratio (PSNR) is a metric used to compare a degraded image and its enhanced version. The higher the PSNR, the better the quality of the reconstruction of the degraded image.

## Model List

 Model name                                             | Architecture | Training Dataset | PSNR FP32                            | PSNR INT8          | Input size | OPS  | Params | FP32 Size | INT8 Size | Compatibility
--------------------------------------------------------|--------------|------------------|-------------------------------------|-------------------|------------|------|--------|-----------|-----------|------------------------------------------
 [Fast-SRGAN](./Fast-SRGAN/README.md)           | SRGAN[2]       |  DIV2k      | 26.23 | 26.13              | All (pref. 128x128)   | 46M | 168K   | 636KB    | 240KB     | i.MX 8M Plus, i.MX 93


## References

[1] Agustsson, E., Timofte, R.: Ntire 2017 "challenge on single image super-resolution: Dataset and study." In: CVPRW. (2017)

[2] Ledig, C., Theis, L., Husz´ar, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al.: "Photo-realistic single image super-resolution using a generative adversarial network." In: CVPR. (2017)