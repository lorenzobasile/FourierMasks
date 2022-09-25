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

from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask

#parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--classornorm', type=str, default="class", help="id for fixed class or fixed norm")
parser.add_argument('--class', type=str, default="0", help="class")
parser.add_argument('--norm', type=str, default="infty", help="norm")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')

args = parser.parse_args()

if args.classornorm=="class":
    num_files=0
    for norm in ["1", "2", "infty"]:
        print(norm)
        path="./single/FMN/"+norm+"/"+args.class

        num_files+=len([f for f in os.listdir(path+"masks")if os.path.isfile(os.path.join(path+"masks", f)) ])

    masks=np.zeros((num_files,128,128))
    labels=np.empty((num_files), dtype="object")
    print(masks.shape)
    i=0
    for norm in ["1", "2", "infty"]:
        path="./single/FMN/"+norm+"/"+args.class
        for mask in os.listdir(path+"masks"):
            masks[i]=np.load(path+"masks/"+mask)
            labels[i]=norm
    masks=masks.reshape(-1,128*128)
    id=[]
    for repetition in range(200):
        data = Data(np.random.choice(masks, 500, replace=False))
        id.append(data.compute_id_2NN(maxk=3))
        print(id)
else:
    num_files=0
    for class in range(10):
        print(class)
        path="./single/FMN/"+args.norm+"/"+str(class)

        num_files+=len([f for f in os.listdir(path+"masks")if os.path.isfile(os.path.join(path+"masks", f)) ])

    masks=np.zeros((num_files,128,128))
    labels=np.empty((num_files))
    print(masks.shape)
    i=0
    for class in range(10):
        path="./single/FMN/"+args.norm+"/"+str(class)
        for mask in os.listdir(path+"masks"):
            masks[i]=np.load(path+"masks/"+mask)
            labels[i]=class
    masks=masks.reshape(-1,128*128)
    id=[]
    for repetition in range(200):
        data = Data(np.random.choice(masks, 2000, replace=False))
        id.append(data.compute_id_2NN(maxk=3))
        print(id)
