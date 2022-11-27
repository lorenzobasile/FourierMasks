import os
import numpy as np
import sys
from Hidalgo.python.dimension import hidalgo


np.set_printoptions(threshold=sys.maxsize)
num_files=0
masks=np.zeros((10*200,128,128), dtype=np.float64)
labels=np.zeros(10*200)

i=0
for cat in range(10):
    path="./single/FMN/1/"+str(cat)
    for k, mask in enumerate(os.listdir(path+"/masks")):
        if k==200:
            break
        masks[i]=np.load(path+"/masks/"+mask)
        labels[i]=cat
        i+=1

print(masks.shape)
masks=masks.reshape(-1,128*128)
id=[]
for repetition in range(1):
    model=hidalgo(K=3, Niter=100000)
    model.fit(masks)
    print(model.d_, model.p_)
    for cat in range(10):
        print(model.Z[200*cat:200*(cat+1)])
