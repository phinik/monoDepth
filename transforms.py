import cv2
import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch


class Pad2Square:
    def __init__(self, shape):
        self._shape = shape

    def __call__(self, data):
        img, label, mask_eval = data[0], data[1], data[2]

        # pad bbox to standard shape because pytorch complains otherwise
        pad_1l = (self._shape[0] - img.shape[0]) // 2
        pad_1r = (self._shape[0] - img.shape[0]) - pad_1l
        pad_2l = (self._shape[1] - img.shape[1]) // 2
        pad_2r = (self._shape[1] - img.shape[1]) - pad_2l

        new_img = np.pad(img, ((pad_1l, pad_1r), (pad_2l, pad_2r), (0, 0)))
        new_label = np.pad(label, ((pad_1l, pad_1r), (pad_2l, pad_2r), (0, 0)))
        new_mask_eval = np.pad(mask_eval, ((pad_1l, pad_1r), (pad_2l, pad_2r), (0, 0)))

        return new_img, new_label, new_mask_eval


class Resize:
    def __init__(self, shape):
        self._shape = shape

    def __call__(self, data):
        img, label, mask_eval = data[0], data[1], data[2]

        img = cv2.resize(img,
                          self.get_resized_image_size(img, max(self._shape)),
                          interpolation=cv2.INTER_LINEAR
                          )

        label = cv2.resize(label,
                          self.get_resized_image_size(label, max(self._shape)),
                          interpolation=cv2.INTER_NEAREST
                          )
            
        mask_eval = cv2.resize(mask_eval,
                          self.get_resized_image_size(mask_eval, max(self._shape)),
                          interpolation=cv2.INTER_NEAREST)
        
        if len(img.shape) < 3:
            img = np.expand_dims(img, 2)

        label = np.expand_dims(label, 2)
        mask_eval = np.expand_dims(mask_eval, 2)

        return img, label, mask_eval

    @staticmethod
    def get_resized_image_size(img, max_dim: int):
        # 0: h, 1: w
        if img.shape[0] >= img.shape[1]:  # h >= w
            return int(img.shape[1] / img.shape[0] * max_dim), max_dim
        else:  # h < w
            return max_dim, int(img.shape[0] / img.shape[1] * max_dim)


class ToTensor:
    def __init__(self):
        self._transform = transforms.ToTensor()

    def __call__(self, data):
        img, label, mask_eval = data[0], data[1], data[2]

        label = torch.from_numpy(label)
        label = torch.moveaxis(label, 2, 0)

        mask_eval = torch.tensor(mask_eval, requires_grad=False)
        mask_eval = torch.moveaxis(mask_eval, 2, 0)
        mask_eval.requires_grad_(False)

        return self._transform(img), label, mask_eval


class RandomQuadraticCrop:
    def __init__(self, size: int):
        self._size = size

    def __call__(self, data):
        img, label, mask_train, mask_eval = data[0], data[1], data[2], data[3]

        shape_0 = img.shape[0]
        shape_1 = img.shape[1]

        x = np.random.randint(150, 550)#np.random.randint(0, shape_0 - self._size)
        y = 550#np.random.randint(0, shape_1 - self._size)

        img = img[x:x+self._size, y:y+self._size, ...]
        label = label[x:x+self._size, y:y+self._size, ...]
        mask_train = mask_train[x:x+self._size, y:y+self._size, ...]
        mask_eval = mask_eval[x:x+self._size, y:y+self._size, ...]

        return img, label, mask_train, mask_eval

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, data):
        img, label, mask_train, mask_eval = data[0], data[1], data[2], data[3]

        active = np.random.choice([1, 0], p=[self._p, 1-self._p])     
        if active:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)
            mask_train = np.flip(mask_train, axis=1)
            mask_eval = np.flip(mask_eval, axis=1)
        
        return img, label, mask_train, mask_eval


class RandomColorJitter:
    def __init__(self, hue: float = 0, saturation: float = 0, vibrance: float = 0):
        self._hue = hue
        self._saturation = saturation
        self._vibrance = vibrance

    def __call__(self, data):
        img, label, mask_eval = data[0], data[1], data[2]

        # convert to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # draw scaling factors
        factor_h = np.random.uniform(max(0, 1-self._hue), max(1, 1 + self._hue)) 
        factor_s = np.random.uniform(max(0, 1-self._saturation), max(1, 1 + self._saturation)) 
        factor_v = np.random.uniform(max(0, 1-self._vibrance), max(1, 1 + self._vibrance)) 

        # scale channels
        img_hsv[..., 0] = img_hsv[..., 0] * factor_h
        img_hsv[..., 1] = img_hsv[..., 1] * factor_s
        img_hsv[..., 2] = img_hsv[..., 2] * factor_v

        # clip channels as scaling with factors > 1 can lead to too big pixel values
        img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0, 180)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2], 0, 255)

        # convert back to BGR
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return img, label, mask_eval
    

class RandomCrop:
    def __init__(self, p: float = .1, min_size: int = 100):
        self._p = p
        self._min_size = min_size

    def __call__(self, data):
        img, label, mask_eval = data[0], data[1], data[2]

        active = np.random.choice([1, 0], p=[self._p, 1-self._p])
        if active:
            # height and width of crop
            h = max(self._min_size, np.random.choice(range(img.shape[0])))
            w = max(self._min_size, np.random.choice(range(img.shape[1])))

            # coordinates of upper left corner of crop
            c_h = np.random.choice(range(0, img.shape[0] - h))
            c_w = np.random.choice(range(0, img.shape[1] - w))            

            # apply crop
            img = img[c_h:c_h+h, c_w:c_w+w, :]
            label = label[c_h:c_h+h, c_w:c_w+w, :]
            mask_eval = mask_eval[c_h:c_h+h, c_w:c_w+w, :]

        return img, label, mask_eval


class RandomRotation:
    def __init__(self, max_deg: float = 0):
        self._max_deg = max_deg

    def __call__(self, data):
        angle = np.random.uniform(-self._max_deg, self._max_deg)

        # TODO





