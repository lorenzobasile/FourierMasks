import foolbox
from utils import ADVtrain, singleAdv, singleInv
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

parser.add_argument('--attack', type=str, default="PGD", help="attack type")
parser.add_argument('--model', type=str, default="resnet", help="model architecture")
args = parser.parse_args()

lam=0.01
data='./data/imagenette2-320/'
model_name=args.model
attack='FMN'
norm='infty'
batch_size=4

pathAdv="./singleAdv/"+attack+"/"+model_name+"/"
pathInv="./singleInv/"+attack+"/"+model_name+"/"
if not os.path.exists(pathAdv) or not os.path.exists(pathInv):
    for i in range(10):
        os.makedirs(pathAdv+"figures/"+str(i), exist_ok=True)
        os.makedirs(pathAdv+"masks/"+str(i), exist_ok=True)
        os.makedirs(pathInv+"figures/"+str(i), exist_ok=True)
        os.makedirs(pathInv+"masks/"+str(i), exist_ok=True)

dataloaders = get_dataloaders(data_dir=data, train_batch_size=batch_size, test_batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_name=='resnet':
    base_model = models.resnet18(weights='IMAGENET1K_V1')
    base_model.fc = torch.nn.Linear(512, 10)
elif model_name=='vit':
    base_model = models.vit_b_16(weights='IMAGENET1K_V1')
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

    idxAdv=singleAdv(modelsAdv, base_model, x,  xadv, y, 1000, optimizersAdv, lam, idxAdv, pathAdv)
    idxInv=singleInv(modelsInv, base_model, x,  xadv, y, 1000, optimizersInv, lam, idxInv, pathInv)
