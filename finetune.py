import torch
import argparse
from data import get_dataloaders
import torchvision.models as models
from utils import train
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resnet", help="model architecture")
args = parser.parse_args()

model_name=args.model_name
data='./data/imagenette2-320/'
batch_size=128
lr=0.01
epochs=20

dataloaders = get_dataloaders(data_dir=data, train_batch_size=batch_size, test_batch_size=batch_size, shuffle_train=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('trained_models/'+model_name):
        os.makedirs('trained_models/'+model_name)

print(f'\nTraining {model_name} model...')
if model_name=='resnet':
    model = models.resnet18(weights='IMAGENET1K_V1')

    model.fc = torch.nn.Linear(512, 10)

    for n,p in model.named_parameters():
        if n!="fc.weight" and n!="fc.bias":
            p.requires_grad=False
        else:
            print(n)
elif model_name=='vit':
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = torch.nn.Linear(512, 10)

    for n,p in model.named_parameters():
        if n!="heads.weight" and n!="heads.bias":
            p.requires_grad=False
        else:
            print(n)


model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            epochs=epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1
        )

train(model, dataloaders, epochs, optimizer)
torch.save(model.state_dict(), "trained_models/"+ model_name + "/clean.pt")
