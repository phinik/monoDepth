import cv2
from os.path import join, splitext, exists
from os import listdir
from typing import List
from torch.utils.data import Dataset
import numpy as np
import pickle

class FeldraumDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()

        self._img_dir = self._get_image_dir(dataset)
        self._labels_dir = self._get_labels_dir(dataset)
        self._files = self._get_filenames()
        
        self._transform = transform       

    def _get_image_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/17vahl/feldraum_depth/train/images"
        else:
            return "/srv/ssd_nvm/17vahl/feldraum_depth/test/images"
        
    def _get_labels_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/17vahl/feldraum_depth/train/depth"
        else:
            return "/srv/ssd_nvm/17vahl/feldraum_depth/test/depth"

    def _get_filenames(self) -> List[str]:
        return listdir(self._img_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        filename = splitext(self._files[idx])[0]
        labelname = filename + "_depth_raw.npz"

        img = cv2.imread(join(self._img_dir, self._files[idx]))  # HWC

        depth = np.load(join(self._labels_dir, labelname))["arr_0"].astype(np.double)
        depth = np.rot90(depth, k=3)
        depth = np.fliplr(depth)
       
        mask = np.where(depth > 0.001, 1, 0)
        mask = np.expand_dims(mask, 2)

        depth = np.clip(depth, 0, 10) / 10
        depth = np.expand_dims(depth, 2)

        data = (img, depth, mask)

        if self._transform:
            data = self._transform(data)

        return {
            "img": data[0],
            "depth": data[1], 
            "mask": data[2],
        }
