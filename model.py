import torch.nn as nn
import torch
import timm


def Classifier(modelname):
    base_model = timm.create_model(modelname, pretrained=True, num_classes=10)
    base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    return base_model

class Mask(nn.Module):

    def __init__(self, mask_size: tuple = (128, 128)):
        super().__init__()
        assert len(mask_size)==2, "mask_size must be a 2-dim tuple, e.g., (128, 128)"

        kernel = torch.ones((1, 1, *mask_size))
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
