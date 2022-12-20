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
    return x
def remove_fund(x):
    x[112,112]=0.0
    return x


parser = argparse.ArgumentParser()
parser.add_argument('--attack1', type=str, default="FMN", help="attack type")
parser.add_argument('--type1', type=str, default="adv", help="mask type")
parser.add_argument('--model1', type=str, default="resnet18", help="network")
parser.add_argument('--attack2', type=str, default="FMN", help="attack type")
parser.add_argument('--type2', type=str, default="adv", help="mask type")
parser.add_argument('--model2', type=str, default="resnet18", help="network")


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

folder1='singleInvEarly'
folder2='singleAdvEarly'

path1="./"+folder1+"/"+attack1+"/"+model1+"/masks/"
path2="./"+folder2+"/"+attack2+"/"+model2+"/masks/"

max_files=min(sum([len(os.listdir(path1+str(i))) for i in range(10)]), sum([len(os.listdir(path2+str(i))) for i in range(10)]))
masks1=np.zeros((max_files,3,224,224), dtype=np.float64)
masks2=np.zeros((max_files,3,224,224), dtype=np.float64)
mul=np.zeros((max_files,3,224,224), dtype=np.float64)
mul_w=np.zeros((max_files,3,224,224), dtype=np.float64)
labels=np.zeros(max_files)
i=0
for c in range(10):
    print(c)
    list1=sorted(os.listdir(path1+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    list2=sorted(os.listdir(path2+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    intersection=[x for x in list1 if x in list2]
    for mask in intersection:
        masks1[i]=np.load(path1+str(c)+"/"+mask)
        masks2[i]=np.load(path2+str(c)+"/"+mask)
        if False:
            mul[i]=masks1[i]*masks2[i]
            mul_w[i]=masks1[i]*masks2[i-1]
        labels[i]=c
        i+=1
masks1=masks1[:i]
masks2=masks2[:i]
labels=labels[:i]

masks1=np.sum(masks1, axis=1)
print(masks1.shape)
masks2=np.sum(masks2, axis=1)

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
full_data=np.concatenate([np.random.permutation(masks1), masks2], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])

print("Using scalar product:")

print("Autocorrelation of first set:")
norm1=np.linalg.norm(masks1, 2, axis=1)
dp=np.sum(masks1*masks1, axis=1)
corrs=np.divide(np.divide(dp, norm1), norm1)
print(np.mean(corrs))



print("Autocorrelation of second set:")
norm2=np.linalg.norm(masks2, 2, axis=1)
dp=np.sum(masks2*masks2, axis=1)
corrs=np.divide(np.divide(dp, norm2), norm2)
print(np.mean(corrs))

print("Correlation (no shuffle):")
dp=np.sum(masks1*masks2, axis=1)
corrs=np.divide(np.divide(dp, norm1), norm2)
print(corrs)
print(masks1[np.argmin(corrs)].sum(), masks2[np.argmin(corrs)].sum())
print("Mean: ", np.mean(corrs))
print("Std: ", np.std(corrs))
for i in range(10):
    print(np.mean(corrs[np.where(labels==i)[0]]))

'''
for i in range(len(masks1)):
    same_label=np.where(labels==labels[i])[0]
    diff_label=np.where(labels!=labels[i])[0]
    max_same=0
    max_diff=0
    for m in same_label:
        corr=np.sum(masks1[i]*masks2[m])/np.linalg.norm(masks1[i], 2)/np.linalg.norm(masks2[m], 2)
        if corr>max_same:
            max_same=corr
    for m in diff_label:
        corr=np.sum(masks1[i]*masks2[m])/np.linalg.norm(masks1[i], 2)/np.linalg.norm(masks2[m], 2)
        if corr>max_diff:
            max_diff=corr
    print(max_same, max_diff)


'''

print("Correlation (shuffle):")
correlations=[]
for i in range(10):
    permuted1=np.random.permutation(masks1)
    norm1=np.linalg.norm(permuted1, 2, axis=1)
    dp=np.sum(permuted1*masks2, axis=1)
    corrs=np.divide(np.divide(dp, norm1), norm2)
    correlations.append(np.mean(corrs))
correlations=np.array(correlations)
print("Mean (of means): ", np.mean(correlations))
print("Std (of means): ", np.std(correlations))

'''
print("Correlation (shuffle):")
permuted1=np.random.permutation(masks1)
norm1=np.linalg.norm(permuted1, 2, axis=1)
dp=np.sum(permuted1*masks2, axis=1)
corrs=np.divide(np.divide(dp, norm1), norm2)
print("Mean: ", np.mean(corrs))
print("Std: ", np.std(corrs))
'''
