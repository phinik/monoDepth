import torch
import torch.nn as nn

import kornia

class L1Loss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._loss = nn.L1Loss(reduction='none')
        self._reduction = reduction

    def forward(self, outputs, labels, mask):
        loss = self._loss(outputs, labels) #/ (labels + 1e-16)
        loss = loss * mask

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss


class MSELoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._reduction = reduction
    
    def forward(self, outputs, labels, mask):
        loss = mask * torch.pow((outputs - labels) , 2) #/ (labels + 1e-16)

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss


class SmoothnessLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"
        self._reduction = reduction

    def forward(self, outputs, labels, mask):
        d_labels = kornia.filters.sobel(labels)
        d_outputs = kornia.filters.sobel(outputs)
        
        loss = mask * torch.abs(d_labels - d_outputs) # / d_labels

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss


class SSIMLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._loss = kornia.losses.SSIMLoss(window_size=5)
        self._reduction = reduction
    
    def forward(self, outputs, labels, mask):
        loss = self._loss(outputs, labels) * mask

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss
        
class berHuLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._reduction = reduction
    
    def forward(self, outputs, labels, mask):
        l1 = torch.abs(outputs - labels) * mask / (labels + 1e-16)

        c = 0.2 * torch.max(l1).item() + 1e-16

        loss = torch.where(l1 <= c, l1, (torch.pow(l1, 2) + c**2) / (2*c))

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss
        
class HuberLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._loss = nn.HuberLoss(reduction='none', delta=0.1)
        self._reduction = reduction
    
    def forward(self, outputs, labels, mask):
        loss = mask * self._loss(outputs, labels)

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss
        
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._loss = nn.SmoothL1Loss(reduction='none', beta=0.1)
        self._reduction = reduction
    
    def forward(self, outputs, labels, mask):
        loss = mask * self._loss(outputs, labels)

        if self._reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        else:
            return loss
        

class ZoeDepthLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()

        assert reduction in {"none", "mean"}, "unsupported reduction"

        self._reduction = reduction
        self._eps = 1e-16

    def forward(self, predictions, ground_truths, masks):        
        d = predictions - ground_truths  # Log yielded worse results
        n = torch.sum(masks, (1, 2, 3))

        term_1 = self._compute_term_1(d, n)
        term_2 = self._compute_term_2(d, n)
        term_3 = self._compute_term_3(d, n)        

        total_loss = term_1 + term_2 + term_3

        if self._reduction == "mean":
            return torch.sum(total_loss) / total_loss.shape[0]
        else:
            return total_loss
        
    def _compute_term_1(self, d, n):
        return torch.sum(torch.pow(d, 2), (1, 2, 3)) / n
    
    def _compute_term_2(self, d, n):
        return torch.pow(torch.sum(d, (1, 2, 3)), 2) / (2 * torch.pow(n, 2))
    
    def _compute_term_3(self, d, n):
        gradients = kornia.filters.spatial_gradient(d)
        grad_x = gradients[:, :, 0, ...]
        grad_y = gradients[:, :, 1, ...]

        return torch.sum(torch.pow(grad_x, 2) + torch.pow(grad_y, 2), (1, 2, 3)) / n


class TotalLoss(nn.Module):   
    def __init__(self, weight_dis: float = 1, weight_smooth: float = 1):
        super().__init__()

        self._distance_loss = berHuLoss("mean") #MSELoss("mean")
        self._smoothness_loss = SmoothnessLoss("mean") #kornia.losses.InverseDepthSmoothnessLoss()

        self._w_distance_loss = weight_dis
        self._w_smoothness_loss = weight_smooth
    
    def forward(self, outputs, labels, mask):
        distance = self._distance_loss(outputs, labels, mask)
        smoothness = self._smoothness_loss(outputs, labels, mask)
    
        return self._w_distance_loss * distance \
            + self._w_smoothness_loss * smoothness \
    
