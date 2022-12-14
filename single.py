import foolbox
from utils import singleAdv, singleInv, singleAttack
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

lam=0.001
data='./data/imagenette2-320/'
model_name=args.model
attack=args.attack
norm='infty'
batch_size=16

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
    base_model = models.resnet101(weights='IMAGENET1K_V1')
    base_model.fc = torch.nn.Linear(2048, 10)
elif model_name=='vit':
    base_model = models.vit_b_16(weights='IMAGENET1K_V1')
    base_model.heads = torch.nn.Linear(768, 10)
elif model_name=='resnet18':
    base_model = models.resnet18(weights='IMAGENET1K_V1')
    base_model.fc = torch.nn.Linear(512, 10)


base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/"+model_name+"/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(0.0, 1.0))

print("Model:", model_name)

adv_dataloaders={'test': DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['test'], 'test'), batch_size=batch_size, shuffle=False)}

idxAdv=0
idxInv=0

for x, xadv, y in adv_dataloaders['test']:
    idxAdv=singleAdv(base_model, x,  xadv, y, 500, lam, idxAdv, pathAdv)
    idxInv=singleInv(base_model, x,  xadv, y, 500, lam, idxInv, pathInv)
    #idxInv=singleAttack(base_model, x,  xadv, y, 500, lam, idxInv, pathInv)
