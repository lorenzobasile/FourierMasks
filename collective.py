import foolbox
from utils import ADVtrain
import timm
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask

parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
parser.add_argument('--norm', type=str, default="infty", help="norm")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

args = parser.parse_args()

eps=args.epsilon
norm=args.norm

dataloaders = get_dataloaders(data_dir=args.data, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(-np.inf,np.inf))

adv_dataloaders = {'train': DataLoader(AdversarialDataset(fmodel, args.attack, dataloaders['train'], 'train', eps, norm), batch_size=args.train_batch_size, shuffle=True),
                   'test': DataLoader(AdversarialDataset(fmodel, args.attack, dataloaders['test'], 'test', eps, norm), batch_size=args.test_batch_size, shuffle=False)}



print("Accuracy evaluation on adversarial test set")
correct=0
correct_adv=0
for x, x_adv, y in adv_dataloaders['test']:
    x=x.to(device)
    x_adv=x_adv.to(device)
    y=y.to(device)
    out = base_model(x)
    out_adv = base_model(x_adv)
    correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
    correct += (torch.argmax(out, axis=1) == y).sum().item()
print(f"Clean Accuracy on test set: {correct / len(adv_dataloaders['test'].dataset) * 100:.5f} %")
print(f"Adversarial Accuracy on test set: {correct_adv / len(adv_dataloaders['test'].dataset) * 100:.5f} %")

lam=1e-2
ne=(str(args.epsilon) if args.attack=="PGD" else str(args.norm))


if not os.path.exists("trained_models/"+args.attack+"/"+ne):
        os.makedirs("trained_models/"+args.attack+"/"+ne)

if not os.path.exists("figures/"+args.attack+"/"+ne):
        os.makedirs("figures/"+args.attack+"/"+ne)


print("L1 penalty: ", lam)

m=Mask().to(device)
model=MaskedClf(m, base_model)

for p in model.clf.parameters():
    p.requires_grad=False

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1
        )

clean, adv, penalties=ADVtrain(model, args.attack, adv_dataloaders, args.epochs, optimizer, lam, hybrid=True, scheduler=scheduler)
torch.save(model.state_dict(), "trained_models/"+args.attack+"/"+str(args.epsilon)+"/lambda_"+str(lam)+".pt")

plt.figure()
plt.semilogy(penalties)
plt.savefig("figures/"+args.attack+"/"+ne+"/penalty.png")

plt.figure()
plt.title("Accuracy during mask training")
plt.plot(clean, label='clean')
plt.plot(adv, label='adversarial')
plt.legend()
plt.savefig("figures/"+args.attack+"/"+ne+"/accuracy.png")
