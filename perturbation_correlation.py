#from dadapy.data import Data
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
    return x

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
    folder='singleAdvEarly'
elif args.type=='inv':
    folder='singleInvEarly'
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

masks=normalize(masks)
perturb=normalize(perturb)
'''
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
full_data=np.concatenate([np.random.permutation(masks), perturb], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])
'''

print("Using scalar product:")

print("Autocorrelation of masks:")
norm1=np.linalg.norm(masks, 2, axis=1)
dp=np.sum(masks*masks, axis=1)
corrs=np.divide(np.divide(dp, norm1), norm1)
print(np.mean(corrs))



print("Autocorrelation of perturbations:")
norm2=np.linalg.norm(perturb, 2, axis=1)
dp=np.sum(perturb*perturb, axis=1)
corrs=np.divide(np.divide(dp, norm2), norm2)
print(np.mean(corrs))

print("Correlation (no shuffle):")
dp=np.sum(masks*perturb, axis=1)
corrs=np.divide(np.divide(dp, norm1), norm2)
print("Mean: ", np.mean(corrs))
print("Std: ", np.std(corrs))

print("Correlation (shuffle):")
correlations=[]
for i in range(10):
    permuted=np.random.permutation(masks)
    norm1=np.linalg.norm(permuted, 2, axis=1)
    dp=np.sum(permuted*perturb, axis=1)
    corrs=np.divide(np.divide(dp, norm1), norm2)
    correlations.append(np.mean(corrs))
correlations=np.array(correlations)
print("Mean (of means): ", np.mean(correlations))
print("Std (of means): ", np.std(correlations))
