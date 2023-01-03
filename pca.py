from sklearn.decomposition import PCA
import torch
from dadapy.data import Data

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
    x[:,112,112]=0.0
    return x



batch_size=8

folderInv='singleInvEarly'
folderAdv='singleAdvEarly'

pathInv="./"+folderInv+"/FMN/resnet18/masks/"
pathAdv="./"+folderAdv+"/FMN/resnet18/masks/"

adv_files=[len(os.listdir(pathAdv+str(i))) for i in range(10)]
inv_files=[len(os.listdir(pathInv+str(i))) for i in range(10)]
masksAdv=[[] for _ in range(10)]
masksInv=[[] for _ in range(10)]
for c in range(10):
    listInv=sorted(os.listdir(pathInv+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    for mask in listInv:
        masksInv[c].append(np.load(pathInv+str(c)+"/"+mask))
    masksInv[c]=np.array(masksInv[c])
for c in range(10):
    listAdv=sorted(os.listdir(pathAdv+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
    for mask in listAdv:
        masksAdv[c].append(np.load(pathAdv+str(c)+"/"+mask))
    masksAdv[c]=np.array(masksAdv[c])
    print(c)
    data = Data(masksAdv[c].reshape(-1,3*224*224), maxk=3)
    print(data.compute_id_2NN()[0])
