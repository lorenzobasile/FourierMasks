import os
import torchvision
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import foolbox as fb


def get_dataloaders(data_dir, train_batch_size, test_batch_size, data_transforms=None, shuffle_train=False, shuffle_test=False):

    if data_transforms is None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        }

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'test']}
    dataloaders = {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test)}
    return dataloaders

class AdversarialDataset(Dataset):

    def __init__(self, model, model_name, attack, dataloader, train):
        c="data/adv/"+attack+"/"+model_name+"/clean/"+train+".pt"
        a="data/adv/"+attack+"/"+model_name+"/adv/"+train+".pt"
        l="data/adv/"+attack+"/"+model_name+"/lbl/"+train+".pt"
        if os.path.isfile(c) and os.path.isfile(a) and os.path.isfile(l):
            self.clean_imgs=torch.load(c)
            self.adv_imgs=torch.load(a)
            self.labels=torch.load(l)
            return
        if not os.path.exists("data/adv/"+attack+"/"+model_name+"/clean"):
            os.makedirs("data/adv/"+attack+"/"+model_name+"/clean")
        if not os.path.exists("data/adv/"+attack+"/"+model_name+"/adv"):
            os.makedirs("data/adv/"+attack+"/"+model_name+"/adv")
        if not os.path.exists("data/adv/"+attack+"/"+model_name+"/adv"):
            os.makedirs("data/adv/"+attack+"/"+model_name+"/adv")
        self.clean_imgs=torch.empty(0,3,224,224)
        self.adv_imgs=torch.empty(0,3,224,224)
        self.labels=torch.empty(0, dtype=torch.int64)

        device=model.device
        for k, (x, y) in enumerate(dataloader):
            x=x.to(device)
            y=y.to(device)
            if attack=='PGD':
                adversary = fb.attacks.PGD()
            elif attack=='FMN':
                adversary = fb.attacks.LInfFMNAttack()
            else:
                adversary = None
            if attack=='PGD':
                x_adv, clipped, is_adv = adversary(model, x, y, epsilons=0.01)
            else:
                x_adv, clipped, is_adv = adversary(model, x, y, epsilons=0.01)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))
            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)
        torch.save(self.clean_imgs, c)
        torch.save(self.adv_imgs, a)
        torch.save(self.labels, l)
    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        return self.clean_imgs[idx], self.adv_imgs[idx], self.labels[idx]
