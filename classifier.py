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

path="./singleAdv/"+attack+"/"+model_name+"/masks/"
num_files=sum([len(os.listdir(path+str(i))) for i in range(10)])
masks=np.zeros((num_files,3,224,224), dtype=np.float64)
mask_labels=np.zeros(num_files)
i=0
for c in range(10):
    for mask in sorted(os.listdir(path+str(c)),key=lambda x: int(os.path.splitext(x)[0])):
        masks[i]=np.load(path+str(c)+"/"+mask)
        mask_labels[i]=c
        i+=1

masks=masks.reshape(num_files, -1)[:i]
mask_labels=mask_labels.reshape(num_files)[:i]

torch_masks=torch.tensor(masks).float()
torch_labels=torch.tensor(mask_labels).long()

print(masks.shape)

print(mask_labels.dtype)

dl=DataLoader(torch.utils.data.TensorDataset(torch_masks,torch_labels), batch_size=1, shuffle=False)

class LinearClf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(in_features=3*224*224, out_features=10, bias=True)

    def forward(self, X):
        return self.layer(X)

model=LinearClf().to(device)

epochs=150

loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

representations=np.zeros((i,10))

losses=[]
for epoch in range(epochs):
    print(epoch)
    correct=0
    for k, (x, y) in enumerate(iter(dl)):
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        if epoch==epochs-1:
            representations[k]=out.cpu().detach().numpy()
        correct+=(torch.argmax(out, axis=1)==y).sum()
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(correct)

reducer=umap.UMAP()
#reducer=TSNE()
embedding=reducer.fit_transform(representations)

plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=mask_labels, s=2)
plt.savefig("umap_adv.png")
