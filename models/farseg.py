import ever as er
import torch
from models.segmentation import Segmentation
import numpy as np
from ever.module import fpn

import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# @er.registry.MODEL.register()
class farSeg(er.ERModule):
    def __init__(self, config=None):
        super().__init__(config)

        self.segmentation = Segmentation(self.config.segmenation)
        layers = [nn.Conv2d(self.config.classifier.in_channels, self.config.classifier.out_channels, 3, 1, 1),
                  nn.UpsamplingBilinear2d(scale_factor=self.config.classifier.scale)]
        self.classifier = nn.Sequential(*layers)
    
        # self.detector = get_detector(**self.config.detector)

        # self.init_from_weight_file()

    def forward(self, x1, y=None):

        y1_feature = self.segmentation(x1)

        y1_pred = self.classifier(y1_feature)

        return y1_pred


    def set_default_config(self):
        self.config.update(dict(
            segmenation=dict(
                model_type='farseg',
                backbone=dict(
                    resnet_type='resnet50',
                    # resnet_type='resnext101_32x8d',
                    pretrained=False,
                    freeze_at=0,
                    output_stride=32,
                ),
                head=dict(
                    fpn=dict(
                        in_channels_list=(256, 512, 1024, 2048),
                        out_channels=256,
                        conv_block=fpn.conv_bn_relu_block
                    ),
                    fs_relation=dict(
                        scene_embedding_channels=2048,
                        in_channels_list=(256, 256, 256, 256),
                        out_channels=256,
                        scale_aware_proj=True
                    ),
                    fpn_decoder=dict(
                        in_channels=256,
                        out_channels=256,
                        in_feat_output_strides=(4, 8, 16, 32),
                        out_feat_output_stride=4,
                        classifier_config=None
                    )
                ),
            ),
            classifier=dict(
                in_channels=256,
                out_channels=1,
                scale=4.0
            ),
        ))

    def log_info(self):
        return dict(
            cfg=self.config
        )

if __name__ == '__main__':
    a = torch.ones(1, 3, 512, 512).cuda()
    a1 = torch.ones(1, 3, 512, 512).cuda()
    model = farSeg(config=None).cuda()

    c = model(a,a1)
    print(c[0].shape,c[1].shape)
