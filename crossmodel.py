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
parser.add_argument('--attack', type=str, default="PGD", help="attack type")
args = parser.parse_args()

batch_size=8
attack=args.attack

pathVit="./singleInv/"+attack+"/resnet/masks/"
pathRes="./singleInvAlt/"+attack+"/resnet/masks/"
num_files=min(sum([len(os.listdir(pathVit+str(i))) for i in range(10)]), sum([len(os.listdir(pathRes+str(i))) for i in range(10)]))
masksVit=np.zeros((num_files,3,224,224), dtype=np.float64)
masksRes=np.zeros((num_files,3,224,224), dtype=np.float64)
i=0
for c in range(10):
    listVit=sorted(os.listdir(pathVit+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    listRes=sorted(os.listdir(pathRes+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    inters=[x for x in listVit if x in listRes]
    j=0
    for mask in inters:
        masksVit[i]=np.load(pathVit+str(c)+"/"+mask)
        masksRes[i]=np.load(pathRes+str(c)+"/"+mask)
        #print(mask, " ", i, " class ", c)
        i+=1
masksVit=masksVit[:i]
masksRes=masksRes[:i]

masksVit=masksVit.reshape(i, -1)
masksRes=masksRes.reshape(i, -1)

print(np.mean(np.sum(masksVit*masksVit, axis=1)))
print(np.mean(np.sum(masksRes*masksRes, axis=1)))
print(np.mean(np.sum(masksVit*masksRes, axis=1)))
print(np.mean(np.sum(np.random.permutation(masksVit)*masksRes, axis=1)))

print(masksVit.shape)

print("ID of masks (adv):")
data = Data(masksVit, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of masks (inv):")
data = Data(masksRes, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of both (no shuffle):")
full_data=np.concatenate([masksVit, masksRes], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])

print("ID of both (shuffle):")
np.random.shuffle(masksVit)
full_data=np.concatenate([masksVit, masksRes], axis=1)
data = Data(full_data, maxk=3)
print(data.compute_id_2NN()[0])
