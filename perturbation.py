from dadapy.data import Data
import torch
import argparse
from torch.utils.data import DataLoader
from data import AdversarialDataset
import os
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def normalize(x):
    m=np.mean(x)
    s=np.std(x)
    return x-m/s

parser = argparse.ArgumentParser()
parser.add_argument('--attack', type=str, default="PGD", help="attack type")
parser.add_argument('--model', type=str, default="resnet", help="model architecture")
parser.add_argument('--type', type=str, default="adv", help="mask type")

args = parser.parse_args()

batch_size=8
model_name=args.model
attack=args.attack
adv_dataloaders = {'test': DataLoader(AdversarialDataset(None, model_name, attack, None, 'test'), batch_size=batch_size, shuffle=False)}

if args.type=='adv':
    folder='singleAdv'
elif args.type=='inv':
    folder='singleInv'
elif args.type=='invalt':
    folder='singleInvAlt'

clean_data=adv_dataloaders['test'].dataset.clean_imgs
adv_data=adv_dataloaders['test'].dataset.adv_imgs
labels=adv_dataloaders['test'].dataset.labels
perturbations=adv_data-clean_data

path="./"+folder+"/"+attack+"/"+model_name+"/masks/"
num_files=sum([len(os.listdir(path+str(i))) for i in range(10)])
masks=np.zeros((num_files,3,224,224), dtype=np.float64)
perturb=np.zeros((num_files,3,224,224), dtype=np.float64)
i=0
for c in range(10):
    for mask in sorted(os.listdir(path+str(c)),key=lambda x: int(os.path.splitext(x)[0])):
        masks[i]=np.load(path+str(c)+"/"+mask)
        perturb[i]=np.fft.fftshift(torch.fft.fft2(perturbations[int(mask[:-4])]).abs().numpy())
        i+=1
        print(i, end='\r')

masks=masks.reshape(num_files, -1)[:i]
perturb=perturb.reshape(num_files, -1)[:i]

masks1=normalize(masks1)
masks2=normalize(masks2)

print("Using ID:")
print("ID of masks:")
data = Data(masks, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of perturbations:")
data = Data(perturb, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of both (no shuffle):")
full_data=np.concatenate([masks, perturb], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of both (shuffle):")
full_data=np.concatenate([np.random.perturbation(masks1), masks2], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])

print("Using scalar product:")

print("Autocorrelation of masks:")
print(np.mean(np.sum(masks*masks, axis=1)))

print("Autocorrelation of perturbations:")
print(np.mean(np.sum(perturb*perturb, axis=1)))

print("Correlation (no shuffle):")
print(np.mean(np.sum(masks*perturb, axis=1)))

print("Correlation (shuffle):")
correlations=[]
for i in range(100):
    correlations.append(np.mean(np.sum(np.random.permutation(masks)*perturb, axis=1)))
correlations=np.array(correlations)
print("mean: ", np.mean(correlations))
print("std: ", np.std(correlations))
