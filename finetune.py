import timm
import torch
import argparse
from data import get_dataloaders
from utils import train
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()

model='vgg11'
data='./data/imagenette2-320/'
batch_size=128
lr=0.001
epochs=50

dataloaders = get_dataloaders(data_dir=data, train_batch_size=batch_size, test_batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('trained_models/'+model):
        os.makedirs('trained_models/'+model)

print(f'\nTraining {model} model...')
model = timm.create_model(model, pretrained=True, num_classes=10)

if model=='vgg11':
    model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
'''
elif args.model=='vit_base_patch16_224':
    for p in model.named_parameters():
        if p[0] == 'patch_embed.proj.bias':
            biases = p[1]
        if p[0] == 'patch_embed.proj.weight':
            weights = p[1]
    weights=weights.mean(axis=1).reshape(768,1,16,16)

    model.patch_embed.proj = torch.nn.Conv2d(1, 768, (16, 16), (16, 16))
    model.patch_embed.proj.weight = torch.nn.Parameter(weights)
    model.patch_embed.proj.bias = torch.nn.Parameter(biases)
    model.patch_embed.num_patches = (128 // 16) ** 2
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, model.patch_embed.num_patches + 1, 768))
'''
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            epochs=epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1
        )

train(model, dataloaders, epochs, optimizer, scheduler)
torch.save(model.state_dict(), "trained_models/"+ model + "/clean.pt")
