import foolbox
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

adv_dataloaders={'test': DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['test'], 'test'), batch_size=batch_size, shuffle=False)}
for lam in [0.001]:
    for lr in [0.01]:
        for image in [0,1,2,3,4]:
            print("lam: ", lam, " lr: ", lr, " image: ", image)
            for x, xadv, y in adv_dataloaders['test']:
                masks=np.zeros((5,3*224*224), dtype=np.float64) # 5 runs on image 0
                x=x.to(device)
                xadv=xadv.to(device)
                y=y.to(device)
                losses=[[] for i in range(5)]
                for i in range(5):
                    print(i)
                    modelAdv=MaskedClf(Mask().to(device), base_model)
                    for p in modelAdv.clf.parameters():
                        p.requires_grad=False
                    optimizer=torch.optim.Adam(modelAdv.parameters(), lr=lr)
                    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=1) #useless with gamma=1
                    loss=torch.nn.CrossEntropyLoss()
                    base_out=base_model(x)
                    for epoch in range(500):
                        modelAdv.mask.train()
                        out=modelAdv(x[image])
                        loss1=loss(out,y[image].reshape(1))
                        #diff = loss1-loss(base_out[0].view(1,-1), y[0].reshape(1)) #nikos' way
                        #invariance = torch.exp(diff**2) #nikos' way
                        invariance=loss1 #the other way
                        penalty=modelAdv.mask.weight.abs().sum()
                        l=penalty*lam+invariance
                        losses[i].append(l.item())
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
                        scheduler.step()
                        modelAdv.mask.weight.data.clamp_(0.0)
                        mask=np.fft.fftshift(modelAdv.mask.weight.detach().cpu().reshape(3,224,224))
                    print(penalty)
                    #only at last epoch save figs
                    masks[i]=mask.reshape(-1)
                    print(mask[0][:10,:10])
                    plt.figure()
                    plt.imshow(mask[0]/np.max(mask[0]), cmap='Blues')
                    plt.colorbar()
                    plt.savefig(path+'image_'+str(image)+'_run_'+str(i)+'_R.png')
                    plt.figure()
                    plt.imshow(mask[1]/np.max(mask[1]), cmap='Blues')
                    plt.colorbar()
                    plt.savefig(path+'image_'+str(image)+'_run_'+str(i)+'_G.png')
                    plt.figure()
                    plt.imshow(mask[2]/np.max(mask[2]), cmap='Blues')
                    plt.colorbar()
                    plt.savefig(path+'image_'+str(image)+'_run_'+str(i)+'_B.png')

                #for each pair compute correlation, it should ideally be 1
                for j in range(5):
                    for k in range(5):
                        print(j, k, "\n")
                        print(np.sum(masks[j]*masks[k])/np.linalg.norm(masks[j], 2)/np.linalg.norm(masks[k], 2))
                break
