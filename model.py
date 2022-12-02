import torch.nn as nn
import torch

class Mask(nn.Module):

    def __init__(self, mask_size: tuple = (3, 224, 224)):
        super().__init__()
        assert len(mask_size)==3, "mask_size must be a 3-dim tuple, e.g., (3, 224, 224)"

        kernel = torch.ones((1, *mask_size))
        self.weight = nn.Parameter(kernel)
        nn.init.ones_(self.weight)

    def forward(self, x):
        x = torch.fft.fft2(x)
        x = self.weight * x
        x = torch.fft.ifft2(x).real
        return x

class MaskedClf(nn.Module):
    def __init__(self, mask, clf):
        super().__init__()
        self.mask=mask
        self.clf=clf
    def forward(self, x):
        x=self.mask(x)
        x=self.clf(x)
        return x
