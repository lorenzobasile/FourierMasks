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

for cat1 in range(9):
    for cat2 in range(cat1+1, 10):
        print(cat1, cat2)
        num_files_1=0
        num_files_2=0
        for norm in ["1"]:#, "2", "infty"]:
            print(norm)
            path="./single/FMN/"+norm+"/"+str(cat1)

            num_files_1+=len([f for f in os.listdir(path+"/masks")if os.path.isfile(os.path.join(path+"/masks", f)) ])
        for norm in ["1"]:#, "2", "infty"]:
            print(norm)
            path="./single/FMN/"+norm+"/"+str(cat2)

            num_files_2+=len([f for f in os.listdir(path+"/masks")if os.path.isfile(os.path.join(path+"/masks", f)) ])

        masks_1=np.zeros((num_files_1,128,128), dtype=np.float64)
        masks_2=np.zeros((num_files_2,128,128), dtype=np.float64)
        i=0
        for norm in ["1"]:#, "2", "infty"]:
            path="./single/FMN/"+norm+"/"+str(cat1)
            for mask in os.listdir(path+"/masks"):
                masks_1[i]=np.load(path+"/masks/"+mask)
                i+=1
        i=0
        for norm in ["1"]:#, "2", "infty"]:
            path="./single/FMN/"+norm+"/"+str(cat2)
            for mask in os.listdir(path+"/masks"):
                masks_2[i]=np.load(path+"/masks/"+mask)
                i+=1
        masks_1=masks_1.reshape(-1,128*128)
        masks_2=masks_2.reshape(-1,128*128)

        id=[]
        id_sizes=[]
        for repetition in range(20):
            indices=np.random.choice(range(min(len(masks_1), len(masks_2))), 200, replace=False)
            model=hidalgo()
            masks=np.concatenate((masks_1[indices], masks_2[indices]))
            model.fit(masks)
            ind=np.argsort(model.d_)
            id.append(model.d_[ind])
            id_sizes.append(model.p_[ind])
            #print(model.d_, model.p_)
            #data = Data(masks[indices], maxk=3)
            #id.append(data.compute_id_2NN()[0])
            #print(id)
        id=np.array(id)
        id_sizes=np.array(id_sizes)
        with open("results_2.txt", "a") as file:
            #file.write("ID:")
            #np.savetxt(file, id)
            #np.savetxt(file, id_sizes)
            file.write("Classes: "+str(cat1)+", "+str(cat2))
            file.write("\nMean: ")
            np.savetxt(file, (np.mean(id, axis=0)))
            file.write("\nStd: ")
            np.savetxt(file, np.std(id, axis=0))
            file.write("\nMean: ")
            np.savetxt(file, np.mean(id_sizes, axis=0))
            file.write("\nStd: ")
            np.savetxt(file, np.std(id_sizes, axis=0))
