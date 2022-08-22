# SemiSL-BE
Codebase for the paper 'Semi-supervised Learning for Building Extraction from Remote Sensing Images'

## 0. Introduction
This project is a inference demo for paper **Semi-supervised Learning for Building Extraction from Remote Sensing Images**

We provide 3 [Farseg](https://github.com/Z-Zheng/FarSeg) models trained with the [WHU Building Extraction Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) (Aerial). The backbone is the Resnet-50, and we have different backbone training strategies:
- Pretrained on the ImageNet (Model 1)
- Pretrained on WHU Building Extraction Dataset with the Self-supervised learning approach BYOL (Model 2)
- Pretrained on WHU Building Extraction Dataset with the pixel level Self-supervised learning approach (ours)

After the backbone is trained, the whole Farseg model is tuned with 15% labels of the whole dataset, the reuslts are:

| Methods | IOU | F-score |
|-----------|-----------|------------|
| Model 1 | 83.96 | 91.28   |
| Model 2 |  84.44 | 91.56 |
| Ours | 85.27 | 92.05 |

This illustrates the befinits from pretraining the backbone with the targe domain images instead of the Imagenet.

## 1. How to use

### 1.Donwload the dataset
 [WHU Building Extraction Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) (find the Aerial dataset)

### 2. Clone the source code
`git clone https://github.com/ArthasMil/SemiSL-BE.gitt`

### 3. Donwload the models 
- [Model 1](https://1drv.ms/u/s!As8QwQacjW0gzleBjIwDN3ph5SI8?e=vmvwzl)
- [Model 2](https://1drv.ms/u/s!As8QwQacjW0gzlaBvitTL_b6pjP_?e=zsHLk0)
- [Ours](https://1drv.ms/u/s!As8QwQacjW0gbrVkaUrNplP3_yA?e=OPM2yu)

### 4. Test
Modify the paths in the 'eval.sh' and then run it.
After inference, the results (*.tif) will be listed in the folder **test_result_temp**. 
Run AccuracyExap.py to calculate IoU and F-score.

## Acknowledge

The BYOL and other SSL approaches are modified from [here](https://github.com/lucidrains). Many thanks for the great projects!

