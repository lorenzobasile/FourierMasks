#from dadapy.data import Data
import torch
import argparse
from torch.utils.data import DataLoader
from data import AdversarialDataset
import os
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--attack', type=str, default="PGD", help="attack type")
parser.add_argument('--model', type=str, default="resnet", help="model architecture")

args = parser.parse_args()
path="./figures/"
if not os.path.exists(path):
    for i in range(10):
        os.makedirs(path+str(i), exist_ok=True)
batch_size=8
model_name=args.model
attack=args.attack
adv_dataloaders = {'test': DataLoader(AdversarialDataset(None, model_name, attack, None, 'test'), batch_size=batch_size, shuffle=False)}


clean_data=adv_dataloaders['test'].dataset.clean_imgs
adv_data=adv_dataloaders['test'].dataset.adv_imgs
labels=adv_dataloaders['test'].dataset.labels
perturbations=adv_data-clean_data

pathAdv="./singleAdv/"+attack+"/"+model_name+"/masks/"
pathInv="./singleInv/"+attack+"/"+model_name+"/masks/"
max_files=min(sum([len(os.listdir(pathAdv+str(i))) for i in range(1)]), sum([len(os.listdir(pathInv+str(i))) for i in range(1)]))

masksAdv=np.zeros((max_files,3,224,224), dtype=np.float64)
masksInv=np.zeros((max_files,3,224,224), dtype=np.float64)
f_perturb=np.zeros((max_files,3,224,224), dtype=np.float64)
img=np.zeros((max_files,3,224,224), dtype=np.float64)
adv=np.zeros((max_files,3,224,224), dtype=np.float64)
perturb=np.zeros((max_files,3,224,224), dtype=np.float64)
lbl=np.zeros(max_files, dtype=np.int64)


i=0
for c in range(1):
    listAdv=sorted(os.listdir(pathAdv+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    listInv=sorted(os.listdir(pathInv+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    intersection=[x for x in listAdv if x in listInv]
    for mask in intersection:
        masksAdv[i]=np.load(pathAdv+str(c)+"/"+mask)
        masksInv[i]=np.load(pathInv+str(c)+"/"+mask)
        f_perturb[i]=np.fft.fftshift(torch.fft.fft2(perturbations[int(mask[:-4])]).abs().numpy())
        img[i]=clean_data[int(mask[:-4])].numpy()
        adv[i]=adv_data[int(mask[:-4])].numpy()
        perturb[i]=perturbations[int(mask[:-4])].numpy()
        lbl[i]=labels[int(mask[:-4])]
        i+=1
        print(i, end='\r')

masksAdv=masksAdv[:i]
masksInv=masksInv[:i]
f_perturb=f_perturb[:i]
img=img[:i]
adv=adv[:i]
perturb=perturb[:i]
lbl=lbl[:i]

perturb-=np.min(perturb)
perturb/=np.max(perturb)

for i in range(len(img)):
    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    axs[0, 0].imshow(img[i].transpose((1,2,0)))
    axs[0, 0].set_title('Image')
    axs[0, 1].imshow(adv[i].transpose((1,2,0)))
    axs[0, 1].set_title('Adversarial')
    axs[0, 2].imshow(perturb[i].transpose((1,2,0)))
    axs[0, 2].set_title('Perturbation')
    axs[1, 0].imshow(f_perturb[i,0], cmap='Blues')
    axs[1, 0].set_title('R (F(perturbation))')
    axs[1, 1].imshow(f_perturb[i,1], cmap='Blues')
    axs[1, 1].set_title('G (F(perturbation))')
    axs[1, 2].imshow(f_perturb[i,2], cmap='Blues')
    axs[1, 2].set_title('B (F(perturbation))')
    axs[2, 0].imshow(masksAdv[i,0], cmap='Blues')
    axs[2, 0].set_title('R (adversarial undo mask)')
    axs[2, 1].imshow(masksAdv[i,1], cmap='Blues')
    axs[2, 1].set_title('G (adversarial undo mask)')
    axs[2, 2].imshow(masksAdv[i,2], cmap='Blues')
    axs[2, 2].set_title('B (adversarial undo mask)')
    axs[3, 0].imshow(masksInv[i,0], cmap='Blues')
    axs[3, 0].set_title('R (class preserv mask)')
    axs[3, 1].imshow(masksInv[i,1], cmap='Blues')
    axs[3, 1].set_title('G (class preserv mask)')
    axs[3, 2].imshow(masksInv[i,2], cmap='Blues')
    axs[3, 2].set_title('B (class preserv mask)')
    plt.savefig(path+str(lbl[i])+"/"+str(i)+".png")
    plt.close()
