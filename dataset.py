from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class Test_datasets(Dataset):
    def __init__(self, path_Data, config):
        super(Test_datasets, self)
        images_list = os.listdir(path_Data+'/test/')
        masks_list = os.listdir(path_Data+'/test/')
        images_list = sorted(images_list)
        masks_list = sorted(masks_list)
        self.data = []
        for i in range(len(images_list)):
            img_path = path_Data+'/test/' + images_list[i]
            mask_path = path_Data+'/test/' + masks_list[i]
            self.data.append([img_path, mask_path])
        self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
        
    