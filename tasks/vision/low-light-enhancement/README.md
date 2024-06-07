# Low-light enhancement

Low-light enhancement is the task of making an image captured under low-light conditions brighter or clearer.


## Datasets

The Low-light enhancement models featured in this Model Zoo are trained on the following datasets.

### LOL

LOL (LOw-Light dataset)[1] is a Low-light enhancement dataset.

It is comprised of 500 low-light and normal-light image pairs and divided into 485 training pairs and 15 testing pairs. All of the images are taken have the same resolution (400x600). The photos represent every-day life scenes and are taken indoors.


## Metrics

### PSNR

Peak signal-to-noise ratio (PSNR) is a metric used to compare a reference image and its enhanced version. The higher the PSNR, the better the quality of the reconstruction of the reference image (ground truth).

## Model List

 Model name                                             | Architecture | Training Dataset | PSNR FP32                            | PSNR INT8          | Input size            | OPS  | Params | FP32 Size | INT8 Size | Compatibility                            
--------------------------------------------------------|--------------|------------------|--------------------------------------|--------------------|-----------------------|------|--------|-----------|-----------|------------------------------------------
 [SCI](./SCI/README.md)                                 | SCI[2]       |  LOL             | 15.11                                | 15.06              | All (Here, 1920x1080) | 752M | 5.87K    | 8KB       | 8KB       | i.MX 8M Plus, i.MX 93      


## References

[1] Wei Chen, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In British Machine Vision Conference, pages 1–12, 2018

[2] Ma, Long and Ma, Tengyu and Liu, Risheng and Fan, Xin and Luo, Zhongxuan. Toward Fast, Flexible, and Robust Low-Light Image Enhancement In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5637-5646, 2022