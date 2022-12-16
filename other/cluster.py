from dadapy.data import Data
import torch
import argparse
from torch.utils.data import DataLoader
#from data import AdversarialDataset
import os
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import umap

parser = argparse.ArgumentParser()
parser.add_argument('--attack', type=str, default="PGD", help="attack type")
parser.add_argument('--model', type=str, default="resnet", help="model architecture")
args = parser.parse_args()

batch_size=8
model_name=args.model
attack=args.attack

path="./singleAdv/"+attack+"/"+model_name+"/masks/"
num_files=sum([len(os.listdir(path+str(i))) for i in range(10)])
masks=np.zeros((num_files,3,224,224), dtype=np.float64)
labels=np.zeros(num_files)
i=0
for c in range(10):
    print(c)
    for mask in sorted(os.listdir(path+str(c)),key=lambda x: int(os.path.splitext(x)[0])):
        masks[i]=np.load(path+str(c)+"/"+mask)
        labels[i]=c
        i+=1

masks=masks.reshape(num_files, -1)
print(masks.shape)
reducer = umap.UMAP()

embedding = reducer.fit_transform(masks)

plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=1)
plt.savefig("umap.png")

print(embedding.shape)

