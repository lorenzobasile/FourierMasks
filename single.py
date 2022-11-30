import foolbox
from utils import ADVtrain, singleAdv, singleInv
import timm
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.models as models

from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask

parser = argparse.ArgumentParser()

parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
parser.add_argument('--norm', type=str, default="infty", help="norm")
parser.add_argument('--train_batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

args = parser.parse_args()

lam=0.01
data='./data/imagenette2-320/'
model='resnet18'
attack='FMN'
norm='infty'
batch_size=4

pathAdv="./singleAdv/"+attack+"/"+norm+"/"
pathInv="./singleInv/"+attack+"/"+norm+"/"
if not os.path.exists(pathAdv):
    for i in range(10):
        os.makedirs(pathAdv+str(i)+"/figures")
        os.makedirs(pathAdv+str(i)+"/masks")
        os.makedirs(pathInv+str(i)+"/figures")
        os.makedirs(pathInv+str(i)+"/masks")

dataloaders = get_dataloaders(data_dir=data, train_batch_size=batch_size, test_batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model = models.resnet18(pretrained=True)
base_model.fc = torch.nn.Linear(512, 10)


base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/"+model+"/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(-np.inf,np.inf))

adv_dataloaders = {'train': DataLoader(AdversarialDataset(fmodel, attack, dataloaders['train'], 'train', norm), batch_size=batch_size, shuffle=False),
                   'test': DataLoader(AdversarialDataset(fmodel, attack, dataloaders['test'], 'test', norm), batch_size=batch_size, shuffle=False)}


idxAdv=0
idxInv=0

for x, xadv, y in adv_dataloaders['test']:
    modelsAdv=[]
    modelsInv=[]
    optimizersAdv=[]
    optimizersInv=[]
    for i in range(batch_size):
        modelAdv=MaskedClf(Mask().to(device), base_model)
        modelInv=MaskedClf(Mask().to(device), base_model)
        for p in modelAdv.clf.parameters():
            p.requires_grad=False
        for p in modelInv.clf.parameters():
            p.requires_grad=False
        modelsAdv.append(modelAdv)
        modelsInv.append(modelInv)
        optimizersAdv.append(torch.optim.Adam(modelAdv.parameters(), lr=0.001))
        optimizersInv.append(torch.optim.Adam(modelInv.parameters(), lr=0.001))

    #idxAdv=singleAdv(modelsAdv, base_model, x,  xadv, y, 1000, optimizersAdv, lam, idxAdv, pathAdv)
    idxInv=singleInv(modelsInv, base_model, x,  xadv, y, 100, optimizersInv, lam, idxInv, pathInv)
