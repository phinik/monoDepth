import torch
import loss


class MAE:
    def __init__(self, n_images):
        self._n_images = n_images

        self._metric = torch.zeros(size=(n_images, 2))
        self._loss = loss.L1Loss(reduction='none')
        self._idx = 0

    def update(self, outputs, labels, mask):
        self._metric[self._idx, 0] = torch.sum(mask)
        self._metric[self._idx, 1] = torch.sum(self._loss(outputs, labels, mask))

        self._idx += 1

    def get(self):
        return torch.sum(self._metric[..., 1]) / torch.sum(self._metric[..., 0])

    def get_per_image(self):
        return self._metric[..., 1] / self._metric[..., 0]


class RMSE:
    def __init__(self, n_images):
        self._n_images = n_images

        self._metric = torch.zeros(size=(n_images, 2))
        self._loss = loss.MSELoss(reduction='none')
        self._idx = 0
    
    def update(self, outputs, labels, mask):
        self._metric[self._idx, 0] = torch.sum(mask)
        self._metric[self._idx, 1] = torch.sum(self._loss(outputs, labels, mask))

        self._idx += 1

    def get(self):
        return torch.sqrt(torch.sum(self._metric[..., 1]) / torch.sum(self._metric[..., 0]))
    
    def get_per_image(self):
        return torch.sqrt(self._metric[..., 1] / self._metric[..., 0])


class FracOfPix:
    def __init__(self, n_images, p: float):
        self._n_images = n_images
        self._p = p

        self._metric = torch.zeros(size=(n_images, 2))
        self._idx = 0
        
    def update(self, outputs, labels, mask):
        self._metric[self._idx, 0] = torch.sum(mask)

        relative_map = torch.maximum(outputs / labels, labels / outputs)
        self._metric[self._idx, 1] = torch.sum(torch.where(relative_map < (1 + self._p), 1, 0) * mask)

        self._idx += 1

    def get(self):
        return torch.sum(self._metric[..., 1]) / torch.sum(self._metric[..., 0])
    
    def get_per_image(self):
        return self._metric[..., 1] / self._metric[..., 0]
