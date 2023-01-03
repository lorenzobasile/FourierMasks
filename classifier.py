import torch
import argparse
from torch.utils.data import DataLoader
from data import AdversarialDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--attack', type=str, default="PGD", help="attack type")
parser.add_argument('--model', type=str, default="resnet", help="model architecture")
args = parser.parse_args()



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=8
model_name=args.model
attack=args.attack
#adv_dataloaders = {'train': DataLoader(AdversarialDataset(None, model_name, attack, None, 'train'), batch_size=batch_size, shuffle=False),
adv_dataloaders = {'test': DataLoader(AdversarialDataset(None, model_name, attack, None, 'test'), batch_size=batch_size, shuffle=False)}

clean_data=adv_dataloaders['test'].dataset.clean_imgs
adv_data=adv_dataloaders['test'].dataset.adv_imgs
labels=adv_dataloaders['test'].dataset.labels
perturbations=adv_data-clean_data

print(labels.dtype)

pathInv="./singleInvEarly/"+attack+"/"+model_name+"/masks/"
num_filesInv=sum([len(os.listdir(pathInv+str(i))) for i in range(10)])
masksInv=np.zeros((num_filesInv,3,224,224), dtype=np.float64)
mask_labelsInv=np.zeros(num_filesInv)
i=0
for c in range(10):
    for mask in sorted(os.listdir(pathInv+str(c)),key=lambda x: int(os.path.splitext(x)[0])):
        masksInv[i]=np.load(pathInv+str(c)+"/"+mask)
        mask_labelsInv[i]=c
        i+=1

masksInv=masksInv.reshape(num_filesInv, -1)[:i]
#masksInv=masksInv[:i]
mask_labelsInv=mask_labelsInv.reshape(num_filesInv)[:i]

pathAdv="./singleAdvEarly/"+attack+"/"+model_name+"/masks/"
num_filesAdv=sum([len(os.listdir(pathAdv+str(i))) for i in range(10)])
masksAdv=np.zeros((num_filesAdv,3,224,224), dtype=np.float64)
mask_labelsAdv=np.zeros(num_filesAdv)
i=0
for c in range(10):
    for mask in sorted(os.listdir(pathAdv+str(c)),key=lambda x: int(os.path.splitext(x)[0])):
        masksAdv[i]=np.load(pathAdv+str(c)+"/"+mask)
        mask_labelsAdv[i]=c
        i+=1


masksAdv=masksAdv.reshape(num_filesAdv, -1)[:i]
#masksAdv=masksAdv[:i]
mask_labelsAdv=mask_labelsAdv.reshape(num_filesAdv)[:i]


torch_masksInv=torch.tensor(masksInv).float()
torch_labelsInv=torch.tensor(mask_labelsInv).long()


torch_masksAdv=torch.tensor(masksAdv).float()
torch_labelsAdv=torch.tensor(mask_labelsAdv).long()


trainSet=torch.utils.data.TensorDataset(torch_masksInv,torch_labelsInv)
trainInv, testInv=torch.utils.data.random_split(trainSet, [0.9, 0.1])

trainInv=DataLoader(trainInv, batch_size=64, shuffle=True)
testInv=DataLoader(testInv, batch_size=1, shuffle=False)
dlAdv=DataLoader(torch.utils.data.TensorDataset(torch_masksAdv,torch_labelsAdv), batch_size=1, shuffle=False)

class LinearClf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv = torch.nn.Conv2d(3,1,1)
        self.layer1 = torch.nn.Linear(in_features=3*224*224, out_features=10, bias=True)
        #self.relu = torch.nn.ReLU()
        #self.layer2 = torch.nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, X):
        #X=self.conv(X).reshape(-1,224*224)
        return self.layer1(X)

model=LinearClf().to(device)

epochs=150

loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

representations=np.zeros((i,10))

losses=[]
for epoch in range(epochs):
    print(epoch)
    correct=0
    total_loss=0
    for k, (x, y) in enumerate(iter(trainInv)):
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        if epoch==epochs-1:
            representations[k]=out.cpu().detach().numpy()
        correct+=(torch.argmax(out, axis=1)==y).sum()
        pen=sum(p.abs().sum() for p in model.parameters())
        l=loss(out, y)+0.00000*pen
        total_loss+=l.item()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("Inv ", correct/len(trainInv.dataset))
    print("Loss ", total_loss)
    correct=0
    for k, (x, y) in enumerate(iter(testInv)):
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        correct+=(torch.argmax(out, axis=1)==y).sum()
    print("Inv test ", correct/len(testInv.dataset))
    correct=0
    perclass=torch.zeros(10)
    for k, (x, y) in enumerate(iter(dlAdv)):
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        num_correct=(torch.argmax(out, axis=1)==y).sum()
        if num_correct==0:
            perclass[y.to('cpu')]+=1
        correct+=num_correct
    print("Adv ", correct/len(dlAdv.dataset))
    print(perclass)
'''
reducer=umap.UMAP()
#reducer=TSNE()
embedding=reducer.fit_transform(representations)

plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=mask_labels, s=2)
plt.savefig("umap_adv.png")
'''
