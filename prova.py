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

lam=0.001
data='./data/imagenette2-320/'
model_name=args.model
attack=args.attack
norm='infty'
batch_size=16

path="./prova/"
if not os.path.exists(path):
    for i in range(10):
        os.makedirs(path+str(i), exist_ok=True)
        
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

#adv_dataloaders = {'train': DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['train'], 'train'), batch_size=batch_size, shuffle=False
adv_dataloaders={'test': DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['test'], 'test'), batch_size=batch_size, shuffle=False)}

idxAdv=0
idxInv=0

for x, xadv, y in adv_dataloaders['test']:
    masks=np.zeros((5,3*224*224), dtype=np.float64)
    x=x.to(device)
    xadv=xadv.to(device)
    y=y.to(device)
    losses=[[] for i in range(5)]
    for i in range(5):
        print(i)
        modelInv=MaskedClf(Mask().to(device), base_model)
        for p in modelInv.clf.parameters():
            p.requires_grad=False
        optimizer=torch.optim.Adam(modelInv.parameters(), lr=0.001)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=1)
        loss=torch.nn.CrossEntropyLoss()
        base_out=base_model(x)
        out_adv=base_model(xadv)
        wereadv=(np.where(torch.logical_and((torch.argmax(base_out, axis=1)==y).cpu(), (torch.argmax(out_adv, axis=1)!=y).cpu()))[0]) #only correctly classified images
        for epoch in range(2000):       
            modelInv.mask.train()
            out=modelInv(x[0])
            loss1=loss(out,y[0].reshape(1))
            diff = loss1-loss(base_out[0].view(1,-1), y[0].reshape(1))
            #pippo = torch.clone(diff)
            invariance = torch.exp(diff**2)
            #print(invariance)
            penalty=modelInv.mask.weight.abs().sum()
            l=penalty*lam+invariance
            print(l)
            losses[i].append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            modelInv.mask.weight.data.clamp_(0.)
            mask=np.fft.fftshift(modelInv.mask.weight.detach().cpu().reshape(3,224,224))
            np.save(path+str(i)+"/"+str(epoch)+".npy", mask)
        print(torch.argmax(out, axis=1))
        masks[i]=mask.reshape(-1)
        plt.figure()
        plt.imshow(mask[0], cmap='Blues')
        plt.colorbar()
        plt.savefig('./testR.png')
        plt.figure()
        plt.imshow(mask[1], cmap='Blues')
        plt.colorbar()
        plt.savefig('./testG.png')
        plt.figure()
        plt.imshow(mask[2], cmap='Blues')
        plt.colorbar()
        plt.savefig('./testB.png')
    for j in range(5):
        for k in range(5):
            print(j, k, "\n")
            print(np.sum(masks[j]*masks[k])/np.linalg.norm(masks[j], 2)/np.linalg.norm(masks[k], 2))
    break
