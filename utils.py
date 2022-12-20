import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MaskedClf, Mask



def train(model, dataloaders, n_epochs, optimizer, scheduler=None):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    for epoch in range(n_epochs):
        print("Epoch: ", epoch+1, '/', n_epochs)
        model.train()
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            out=model(x)
            l=loss(out, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        for i in ['train', 'test']:
            correct=0
            with torch.no_grad():
                for x, y in dataloaders[i]:
                    out=model(x.to(device))
                    correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
            print("Accuracy on "+i+" set: ", correct/len(dataloaders[i].dataset))


def singleAdv(base_model, clean, adv, y, n_epochs, lam, idx, path):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(base_model.parameters()).is_cuda else "cpu")
    n=len(clean)
    adv=adv.to(device)
    y=y.to(device)
    clean=clean.to(device)
    base_out=base_model(clean)
    base_adv=base_model(adv)
    losses=[[] for i in range(n)]
    wereadv=(np.where(torch.logical_and((torch.argmax(base_out, axis=1)==y).cpu(), (torch.argmax(base_adv, axis=1)!=y).cpu()))[0]) #only correctly classified images
    for i in range(n):
        model=MaskedClf(Mask().to(device), base_model)
        for p in model.clf.parameters():
            p.requires_grad=False
        model.mask.train()
        optimizer=torch.optim.Adam(model.mask.parameters(), lr=0.01)
        epoch=0
        while True:
            out=model(adv[i])
            l=loss(out, y[i].reshape(1))
            penalty=model.mask.weight.abs().sum()
            l+=penalty*lam
            losses[i].append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            model.mask.weight.data.clamp_(0.)
            epoch+=1
            if epoch>500 and abs(l.item()-np.mean(losses[i][-20:]))<1e-5:
                model.eval()
                correct = torch.argmax(out, axis=1)==y[i] and torch.argmax(model(clean[i]), axis=1)==y[i]
                if correct and i in wereadv:
                    mask=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(3,224,224))
                    #plt.figure(figsize=(30,20))
                    #plt.plot(losses[i])
                    #plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"loss.png")
                    plt.figure()
                    plt.imshow(mask[0], cmap="Blues")
                    plt.colorbar()
                    plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"R.png")
                    plt.figure()
                    plt.imshow(mask[1], cmap="Blues")
                    plt.colorbar()
                    plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"G.png")
                    plt.figure()
                    plt.imshow(mask[2], cmap="Blues")
                    plt.colorbar()
                    plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"B.png")
                    np.save(path+"masks/"+str(y[i].item())+"/"+str(idx)+".npy", mask)
                idx+=1
                break
    return idx



def singleInv(base_model, clean, x, y, n_epochs, lam, idx, path):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(base_model.parameters()).is_cuda else "cpu")

    n=len(x)
    x=x.to(device)
    y=y.to(device)
    clean=clean.to(device)
    base_out=base_model(clean)
    losses=[[] for i in range(n)]
    werecorrect=(np.where((torch.argmax(base_out, axis=1)==y).cpu())[0]) #only correctly classified images

    for i in range(n):
        model=MaskedClf(Mask().to(device), base_model)
        for p in model.clf.parameters():
            p.requires_grad=False
        model.mask.train()
        optimizer=torch.optim.Adam(model.mask.parameters(), lr=0.01)
        epoch=0
        while True:
            out=model(clean[i])
            invariance=loss(out, y[i].reshape(1))
            penalty=model.mask.weight.abs().sum()
            l=penalty*lam+invariance
            losses[i].append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            model.mask.weight.data.clamp_(0.)
            epoch+=1
            if epoch>500 and abs(l.item()-np.mean(losses[i][-20:]))<1e-5:
                correct=torch.argmax(out, axis=1)==y[i]
                if correct and i in werecorrect:
                    mask=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(3,224,224))
                    #plt.figure(figsize=(30,20))
                    #plt.plot(losses[i])
                    #plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"loss.png")
                    plt.figure()
                    plt.imshow(mask[0], cmap="Blues")
                    plt.colorbar()
                    plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"R.png")
                    plt.figure()
                    plt.imshow(mask[1], cmap="Blues")
                    plt.colorbar()
                    plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"G.png")
                    plt.figure()
                    plt.imshow(mask[2], cmap="Blues")
                    plt.colorbar()
                    plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"B.png")
                    np.save(path+"masks/"+str(y[i].item())+"/"+str(idx)+".npy", mask)
                idx+=1
                break
    return idx
