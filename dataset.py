import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np

class SARToImageDataset(Dataset):
    """SAR to Image Datset"""

    def __init__(self, root_dir, sar_dir, eo_dir):
        self.root_dir = root_dir
        self.sar_dir = sar_dir
        self.eo_dir = eo_dir
        self.sar_files = glob.glob(os.path.join(self.root_dir, self.sar_dir) + '/*.png')
        # self.eo_files = glob.glob(os.path.join(root_dir, eo_dir) + '/.png')

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sar_img_name = self.sar_files[idx].split('/')[-1].split('.png')[0]
        sar_img = io.imread(os.path.join(self.root_dir, self.sar_dir) + '/' +  sar_img_name + '.png')
        sar_img = np.expand_dims(sar_img, axis=0)
        
        eo_img_name = os.path.join(self.root_dir, self.eo_dir) + '/' + sar_img_name + '.png'
        eo_img = io.imread(eo_img_name)
        # Transform into CxHXW Format
        eo_img = np.transpose(eo_img, (2,0,1))
        return (sar_img, eo_img)