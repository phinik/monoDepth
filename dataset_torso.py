import cv2
from os.path import join, splitext, sep
from os import listdir
from typing import List
from torch.utils.data import Dataset
import torch
import numpy as np

class TorsoReal(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()

        self._img_dir = self._get_image_dir(dataset)
        self._files = self._get_filenames()
        
        self._transform = transform       

    def _get_image_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/dataset/TORSO21/reality/train/images"
        else:
            return "/srv/ssd_nvm/dataset/TORSO21/reality/test/images"
        
    def _get_filenames(self) -> List[str]:
        return listdir(self._img_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        filename = self._files[idx]

        img = cv2.imread(join(self._img_dir, filename))  # HWC
        mask = np.ones(shape=(*img.shape[0:2], 1))

        data = (img, mask, mask.copy())

        if self._transform:
            data  = self._transform(data)

        return {
            "img": data[0],
            "depth": data[1], 
            "mask": data[1],
            "domain": torch.tensor([1.])
        }
    
class TorsoSim(Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()

        self._img_dir = self._get_image_dir(dataset)
        self._depth_dir = self._get_depth_dir(dataset)
        self._files = self._get_filenames()
        
        self._transforms = transforms  

    def _get_image_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/train/images"
        else:
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/test/images"
        
    def _get_depth_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/train/depth"
        else:
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/test/depth"
        
    def _get_filenames(self) -> List[str]:
        return listdir(self._img_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        filename = splitext(self._files[idx])[0]

        img = cv2.imread(join(self._img_dir, filename + ".PNG"))  # HWC
        depth = np.load(join(self._depth_dir, filename + "_depth_raw.npz"))['arr_0']
        depth = np.clip(depth, 0, 10) / 10
        depth = np.expand_dims(depth, 2)
        depth = np.swapaxes(depth, 0, 1)

        mask = np.ones(shape=(*img.shape[0:2], 1))

        data = (img, depth, mask)

        if self._transforms:
            data  = self._transforms(data)

        return {
            "img": data[0],
            "depth": data[1],
            "mask": data[2],
            "domain": torch.tensor([0.])
        }

class TorsoSimCycled(Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()

        self._img_dir = self._get_image_dir(dataset)
        self._depth_dir = self._get_depth_dir(dataset)
        self._files = self._get_filenames()
        
        self._transforms = transforms  

    def _get_image_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/21donn/datasets/torso_cycled_20/train/images"
        else:
            return "/srv/ssd_nvm/21donn/datasets/torso_cycled_20/test/images"
        
    def _get_depth_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/train/depth"
        else:
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/test/depth"
        
    def _get_filenames(self) -> List[str]:
        return listdir(self._img_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        filename = splitext(self._files[idx])[0]

        img = cv2.imread(join(self._img_dir, filename + ".png"))  # HWC
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        depth = np.load(join(self._depth_dir, filename + "_depth_raw.npz"))['arr_0']
        depth = np.clip(depth, 0, 10) / 10
        depth = np.expand_dims(depth, 2)
        depth = np.swapaxes(depth, 0, 1)

        mask = np.ones(shape=(*img.shape[0:2], 1))

        data = (img, depth, mask)

        if self._transforms:
            data  = self._transforms(data)

        return {
            "img": data[0],
            "depth": data[1],
            "mask": data[2],
            "domain": torch.tensor([0.])
        }


class TorsoSimAndCycled(Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()

        self._img_dirs = self._get_image_dir(dataset)
        self._depth_dir = self._get_depth_dir(dataset)
        self._files = self._get_filenames()
        
        self._transforms = transforms  

    def _get_image_dir(self, dataset) -> str:
        if dataset == "train":
            return [
                "/srv/ssd_nvm/21donn/datasets/torso_cycled_100/train/images",
                "/srv/ssd_nvm/dataset/TORSO21/simulation/train/images"
            ]
        else:
            return [
                "/srv/ssd_nvm/21donn/datasets/torso_cycled_100/test/images",
                "/srv/ssd_nvm/dataset/TORSO21/simulation/test/images"
            ]
        
    def _get_depth_dir(self, dataset) -> str:
        if dataset == "train":
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/train/depth"
        else:
            return "/srv/ssd_nvm/dataset/TORSO21/simulation/test/depth"
        
    def _get_filenames(self) -> List[str]:
        paths = []
        
        for directory in self._img_dirs:
            files = listdir(directory)

            full_path_files = []
            for file in files:
                full_path_files.append(join(directory, file))

            paths += full_path_files

        return paths

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        filename = splitext(self._files[idx])[0].split(sep)[-1]

        img = cv2.imread(self._files[idx])  # HWC

        if img.shape != (1080, 1920, 3):
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        depth = np.load(join(self._depth_dir, filename + "_depth_raw.npz"))['arr_0']
        depth = np.clip(depth, 0, 10) / 10
        depth = np.expand_dims(depth, 2)
        depth = np.swapaxes(depth, 0, 1)

        mask = np.ones(shape=(*img.shape[0:2], 1))

        data = (img, depth, mask)

        if self._transforms:
            data  = self._transforms(data)

        return {
            "img": data[0],
            "depth": data[1],
            "mask": data[2],
            "domain": torch.tensor([0.])
        }
