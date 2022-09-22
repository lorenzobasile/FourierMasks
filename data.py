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
                transforms.Resize(128),
                transforms.RandomResizedCrop((128,128)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.RandomAffine(degrees=0.),
                transforms.ToTensor(),
                transforms.Normalize((0.449,), (0.226,))
            ]),
            'test': transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop((128,128)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.449,), (0.226,))
            ]),
        }

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'test']}
    dataloaders = {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test)}
    return dataloaders

class AdversarialDataset(Dataset):

    def __init__(self, model, adversarytype, dataloader, train, eps, norm):
        c="data/adv/"+adversarytype+"/"+str(eps)+"/clean/"+train+".pt"
        a="data/adv/"+adversarytype+"/"+str(eps)+"/adv/"+train+".pt"
        l="data/adv/"+adversarytype+"/"+str(eps)+"/lbl/"+train+".pt"
        if os.path.isfile(c) and os.path.isfile(a) and os.path.isfile(l):
            self.clean_imgs=torch.load(c)
            self.adv_imgs=torch.load(a)
            self.labels=torch.load(l)
            return
        if not os.path.exists("data/adv/"+adversarytype+"/"+str(eps)+"/clean"):
            os.makedirs("data/adv/"+adversarytype+"/"+str(eps)+"/clean")
        if not os.path.exists("data/adv/"+adversarytype+"/"+str(eps)+"/adv"):
            os.makedirs("data/adv/"+adversarytype+"/"+str(eps)+"/adv")
        if not os.path.exists("data/adv/"+adversarytype+"/"+str(eps)+"/lbl"):
            os.makedirs("data/adv/"+adversarytype+"/"+str(eps)+"/lbl")
        self.clean_imgs=torch.empty(0,1,128,128)
        self.adv_imgs=torch.empty(0,1,128,128)
        self.labels=torch.empty(0, dtype=torch.int64)

        device=model.device
        for k, (x, y) in enumerate(dataloader):
            x=x.to(device)
            y=y.to(device)
            if adversarytype=='PGD':
                adversary = fb.attacks.PGD()
            elif adversarytype=='FMN':
                if norm=="1":
                    adversary = fb.attacks.L1FMNAttack()
                elif norm=="2":
                    adversary = fb.attacks.L2FMNAttack()
                else:
                    adversary = fb.attacks.LInfFMNAttack()
            else:
                adversary = None
            x_adv, clipped, is_adv = adversary(model, x, y, epsilons=eps)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))
            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)

        '''
        device=torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        for k, (x, y) in enumerate(dataloader):
            print("batch ", k)
            x=x.to(device)
            y=y.to(device)
            if adversarytype=='PGD':
                adversary = PGD(model, 'cuda')
                x_adv = adversary.generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10, clip_min=-np.inf, clip_max=np.inf)
            else:
                adversary = PGD(model, 'cuda')
                x_adv = adversary.generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10, bound='l2', clip_min=-np.inf, clip_max=np.inf)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))
            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)
        '''

        torch.save(self.clean_imgs, c)
        torch.save(self.adv_imgs, a)
        torch.save(self.labels, l)
    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        return self.clean_imgs[idx], self.adv_imgs[idx], self.labels[idx]
