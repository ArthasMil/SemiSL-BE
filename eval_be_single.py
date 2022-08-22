from opt import get_opts
import torch
import os
import warnings
warnings.filterwarnings("ignore")
from utils.Dataloader_WHU import DatasetWHU_WithName
from models.farseg import farSeg
import numpy as np
import torch.nn.functional as F
import scipy.misc
import cv2
# import re

torch.backends.cudnn.benchmark = True # this increases inference speed a little

   
def extract_model_state_dict(ckpt_path, prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        for k, v in checkpoint['state_dict'].items():
            if not k.startswith('model.'):
                continue
            k = k[6:] # remove 'model.'
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    else: # if it only has model weights
        for k, v in checkpoint.items():
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
 
if __name__ == '__main__':
    hparams = get_opts()
    val_images_path = os.path.join(hparams.root_dir, 'test', 'image')
    val_labels_path = os.path.join(hparams.root_dir, 'test', 'label')
    val_dataset = DatasetWHU_WithName(val_images_path, val_labels_path,data_aug=False)
    dataloader_test = torch.utils.data.DataLoader(dataset = val_dataset,
                                                num_workers = 0,
                                                shuffle = True,
                                                pin_memory=True)

    model = farSeg()
    device_cu = 0
    model.cuda(device_cu)
    model.load_state_dict(torch.load(hparams.ckpt_path))
    # load_ckpt(model, hparams.ckpt_path)
    # torch.save(model.state_dict(),'PtImangeNet_best_iou.pth')
    model.eval()
    
    # 开始inference
    for image, _ , imageName in dataloader_test:
        # print(image.shape)
        with torch.no_grad():
            be = model(image.cuda(device_cu))
            predict = F.sigmoid(be)
            # predict = model(image.cuda())
            predict = predict.cpu().detach().numpy()
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 255
        # predict = torch.where(predict>0.5,torch.ones_like(output),torch.zeros_like(output)) * 255
        result = np.squeeze(predict)
        # scipy.misc.imsave('./test_result_temp/{}'.format(imageName[0]), result)
        cv2.imwrite('./test_result_temp/{}'.format(imageName[0]),result)