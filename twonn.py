from dadapy.data import Data
from utils import ADVtrain, single
import timm
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import sys
from scipy.spatial import distance
from Hidalgo.python.dimension import TwoNN
from Hidalgo.python.dimension import hidalgo


from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask
parser = argparse.ArgumentParser()

#parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--classornorm', type=str, default="class", help="id for fixed class or fixed norm")
parser.add_argument('--cat', type=str, default="0", help="class")
parser.add_argument('--norm', type=str, default="infty", help="norm")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')

args = parser.parse_args()
np.set_printoptions(threshold=sys.maxsize)

if args.classornorm=="class":
    num_files=0
    for norm in ["1", "2", "infty"]:
        print(norm)
        path="./single/FMN/"+norm+"/"+args.cat

        num_files+=len([f for f in os.listdir(path+"/masks")if os.path.isfile(os.path.join(path+"/masks", f)) ])

    masks=np.zeros((num_files,128,128), dtype=np.float64)
    labels=np.empty((num_files), dtype="object")
    print(masks.shape)
    i=0
    for norm in ["1", "2", "infty"]:
        path="./single/FMN/"+norm+"/"+args.cat
        for mask in os.listdir(path+"/masks"):
            masks[i]=np.load(path+"/masks/"+mask)
            labels[i]=norm
            i+=1
    masks=masks.reshape(-1,128*128)

    id=[]
    for repetition in range(200):
        indices=np.random.choice(range(len(masks)), 500, replace=False)


        model=hidalgo()
        
        #model=hidalgo(K=2,Niter=2000,zeta=0.65,q=5,Nreplicas=10,burn_in=0.8)
        
        
        model.fit(masks[indices])
        
        print(model.d_,model.derr_)
        print(model.p_,model.perr_)
        #print(model.lik_, model.likerr_)
        #print(model.Pi)
        #print(model.Z)
        '''
        model = TwoNN()
        model.fit(masks[indices])
        print(model.DimEstimate_)
        '''
        
        data = Data(masks[indices], maxk=3)
        id.append(data.compute_id_2NN()[0])
        print(id)
    print("Mean:", np.mean(id))
    print("Std:", np.std(id))
else:
    num_files=0
    for cat in range(10):
        print(cat)
        path="./single/FMN/"+args.norm+"/"+str(cat)

        num_files+=len([f for f in os.listdir(path+"/masks")if os.path.isfile(os.path.join(path+"/masks", f)) ])

    masks=np.zeros((num_files,128,128))
    labels=np.empty((num_files))
    print(masks.shape)
    i=0
    for cat in range(10):
        path="./single/FMN/"+args.norm+"/"+str(cat)
        for mask in os.listdir(path+"/masks"):
            masks[i]=np.load(path+"/masks/"+mask)
            labels[i]=cat
            i+=1
    masks=masks.reshape(-1,128*128)
    id=[]
    for repetition in range(200):
        indices=np.random.choice(range(len(masks)), 500, replace=False)
        data = Data(masks[indices], maxk=3)
        id.append(data.compute_id_2NN()[0])
    print("Mean:", np.mean(id))
    print("Std:", np.std(id))
