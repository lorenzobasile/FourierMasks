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

data='./data/imagenette2-320/'
model_name=args.model
attack=args.attack
batch_size=16

dataloaders = get_dataloaders(data_dir=data, train_batch_size=batch_size, test_batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_name=='resnet':
    base_model = models.resnet101(weights='IMAGENET1K_V1')
    base_model.fc = torch.nn.Linear(2048, 10)
elif model_name=='vit':
    base_model = models.vit_b_16(weights='IMAGENET1K_V1')
    base_model.heads = torch.nn.Linear(768, 10)


base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/"+model_name+"/clean.pt"))
base_model.eval()

print("Model:", model_name)

adv_dataloaders = {'train': DataLoader(AdversarialDataset(None, model_name, attack, dataloaders['train'], 'train'), batch_size=batch_size, shuffle=False),
                   'test': DataLoader(AdversarialDataset(None, model_name, attack, dataloaders['test'], 'test'), batch_size=batch_size, shuffle=False)}

correct=np.zeros(10)
correct_adv=np.zeros(10)
j=0
for x, xadv, y in adv_dataloaders['test']:
    x=x.to(device)
    xadv=xadv.to(device)
    y=y.to(device)
    out_clean=base_model(x)
    out_adv=base_model(xadv)
    corr=(torch.argmax(out_clean, axis=1)==y)
    corr_adv=(torch.argmax(out_adv, axis=1)==y)
    for i in range(len(corr)):
        if corr[i]:
            print(j, " of class ", y[i], " correct")
            correct[y[i]]+=1
        if corr_adv[i]:
            print(j, " of class ", y[i], " correct adv")
            correct_adv[y[i]]+=1
        j+=1
print(correct)
print(correct_adv)
