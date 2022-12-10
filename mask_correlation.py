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
    return (x-m)/s

parser = argparse.ArgumentParser()
parser.add_argument('--attack1', type=str, default="FMN", help="attack type")
parser.add_argument('--type1', type=str, default="adv", help="mask type")
parser.add_argument('--model1', type=str, default="resnet", help="network")
parser.add_argument('--attack2', type=str, default="FMN", help="attack type")
parser.add_argument('--type2', type=str, default="adv", help="mask type")
parser.add_argument('--model2', type=str, default="resnet", help="network")


args = parser.parse_args()

batch_size=8
attack1=args.attack1
attack2=args.attack2
model1=args.model1
model2=args.model2

if args.type1=='adv':
    folder1='singleAdv'
elif args.type1=='inv':
    folder1='singleInv'
elif args.type1=='invalt':
    folder1='singleInvAlt'

if args.type2=='adv':
    folder2='singleAdv'
elif args.type2=='inv':
    folder2='singleInv'
elif args.type2=='invalt':
    folder2='singleInvAlt'

path1="./"+folder1+"/"+attack1+"/"+model1+"/masks/"
path2="./"+folder2+"/"+attack2+"/"+model2+"/masks/"

max_files=min(sum([len(os.listdir(path1+str(i))) for i in range(10)]), sum([len(os.listdir(path2+str(i))) for i in range(10)]))
masks1=np.zeros((max_files,3,224,224), dtype=np.float64)
masks2=np.zeros((max_files,3,224,224), dtype=np.float64)
i=0
for c in range(10):
    list1=sorted(os.listdir(path1+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    list2=sorted(os.listdir(path2+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    intersection=[x for x in list1 if x in list2]
    for mask in intersection:
        masks1[i]=np.load(path1+str(c)+"/"+mask)
        masks2[i]=np.load(path2+str(c)+"/"+mask)
        i+=1
masks1=masks1[:i]
masks2=masks2[:i]

masks1=masks1.reshape(i, -1)
masks2=masks2.reshape(i, -1)

masks1=normalize(masks1)
masks2=normalize(masks2)
print("Using ID:")
print("ID of first set:")
data = Data(masks1, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of second set:")
data = Data(masks2, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of both (no shuffle):")
full_data=np.concatenate([masks1, masks2], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of both (shuffle):")
full_data=np.concatenate([np.random.perturbation(masks1), masks2], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])

print("Using scalar product:")

print("Autocorrelation of first set:")
print(np.mean(np.sum(masks1*masks1, axis=1)))

print("Autocorrelation of second set:")
print(np.mean(np.sum(masks2*masks2, axis=1)))

print("Correlation (no shuffle):")
print(np.mean(np.sum(masks1*masks2, axis=1)))

print("Correlation (shuffle):")
correlations=[]
for i in range(100):
    correlations.append(np.mean(np.sum(np.random.permutation(masks1)*masks2, axis=1)))
correlations=np.array(correlations)
print("mean: ", np.mean(correlations))
print("std: ", np.std(correlations))
