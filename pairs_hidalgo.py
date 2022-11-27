import os
import numpy as np
import sys
from Hidalgo.python.dimension import hidalgo

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
        accuracy=[]
        for repetition in range(20):
            indices=np.random.choice(range(min(len(masks_1), len(masks_2))), 200, replace=False)
            model=hidalgo()
            perm=np.random.permutation(400)
            masks=np.concatenate((masks_1[indices], masks_2[indices]))
            labels=np.concatenate((np.ones(200)*1, np.ones(200)*2))
            alt_labels=np.concatenate((np.ones(200)*2, np.ones(200)*1))
            masks=masks[perm]
            labels=labels[perm]
            alt_labels=alt_labels[perm]
            model.fit(masks)
            ind=np.argsort(model.d_)
            id.append(model.d_[ind])
            id_sizes.append(model.p_[ind])
            correct=np.sum(model.Z==labels)
            accuracy.append(max(correct, 400-correct)/400)
            #print(model.d_, model.p_)
            #data = Data(masks[indices], maxk=3)
            #id.append(data.compute_id_2NN()[0])
            #print(id)
        id=np.array(id)
        id_sizes=np.array(id_sizes)
        accuracy=np.array(accuracy)
        with open("results_2.txt", "a") as file:
            #file.write("ID:")
            #np.savetxt(file, id)
            #np.savetxt(file, id_sizes)
            file.write("\nClasses: "+str(cat1)+", "+str(cat2))
            file.write("\nIDs: ")
            np.savetxt(file, (np.mean(id, axis=0)))
            #file.write("\nStd: ")
            #np.savetxt(file, np.std(id, axis=0))
            file.write("\nID sizes: ")
            np.savetxt(file, np.mean(id_sizes, axis=0))
            #file.write("\nStd: ")
            #np.savetxt(file, np.std(id_sizes, axis=0))
            file.write("\nID accuracy: ")
            file.write(str(np.mean(accuracy)))
