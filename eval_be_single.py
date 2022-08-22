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

torch.backends.cudnn.benchmark = True # this increases inference speed a little

  
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
    model.eval()
    
    for image, _ , imageName in dataloader_test:
        with torch.no_grad():
            be = model(image.cuda(device_cu))
            predict = F.sigmoid(be)
            predict = predict.cpu().detach().numpy()
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 255
        result = np.squeeze(predict)
        cv2.imwrite('./test_result_temp/{}'.format(imageName[0]),result)
