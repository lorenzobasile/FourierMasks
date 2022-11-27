import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

acc=np.zeros((10,10))
with open('results_2.txt', 'r') as file:
    for line in file:
        if len(line.split())>0 and line.split()[0]=='Classes:':
            class1=int(line.split()[1][:-1])
            class2=int(line.split()[2])
        if len(line.split())>1 and line.split()[1]=='accuracy:':
            acc[class1][class2]=line.split()[2]
            acc[class2][class1]=acc[class1][class2]
hm = sns.heatmap(data=acc, annot=True)
plt.show()
