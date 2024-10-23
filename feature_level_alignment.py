import torch.nn as nn
from torch.autograd import Function
import torch 

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
    
class GradOutput(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        print(torch.max(grad_output), torch.min(grad_output))
        return grad_output.view_as(grad_output)
    
class FeatureAlignment(nn.Module):
    def __init__(self, n_input: int):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(n_input, 1024),
                nn.Sigmoid(),
                nn.Linear(1024, 1024),
                nn.Sigmoid(),
                nn.Linear(1024, 1),
                nn.Sigmoid(), #(dim=1),
            )

    def forward(self, x):
        x = GradReverse.apply(x)
        x = self.model(x)
        
        return x
     
