from dadapy.data import Data
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from Hidalgo.python.dimension import hidalgo


from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask
parser = argparse.ArgumentParser()

#parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--classornorm', type=str, default="class", help="id for fixed class or fixed norm")
parser.add_argument('--cat1', type=str, default="0", help="class")
parser.add_argument('--cat2', type=str, default="0", help="class")
parser.add_argument('--norm', type=str, default="infty", help="norm")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')

args = parser.parse_args()
np.set_printoptions(threshold=sys.maxsize)
norm=args.norm
num_files=0
masks=np.zeros((1000,128,128), dtype=np.float64)
labels=np.empty((1000), dtype="object")
    
i=0
for cat in range(10):
    #print(norm)
    path="./single/FMN/"+norm+"/"+str(cat)
    #num_files+=len([f for f in os.listdir(path+"/masks")if os.path.isfile(os.path.join(path+"/masks", f)) ])
    for k, mask in enumerate(os.listdir(path+"/masks")):
        if k==100:
            break
        masks[i]=np.load(path+"/masks/"+mask)
        labels[i]=norm
        i+=1

print(masks.shape)
masks=masks.reshape(-1,128*128)
id=[]
for repetition in range(1):
    #indices=np.random.choice(range(len(masks)), , replace=False)
    model=hidalgo(K=10,Niter=10000)
    model.fit(masks)
    print(model.d_, model.p_)
    '''
    for i in range(10):
        print((model.Z[200*i:200*(i+1)]-np.ones(200)).sum()/200)
    
    data = Data(masks, maxk=3)
    id.append(data.compute_id_2NN()[0])
    print(id)
    '''
