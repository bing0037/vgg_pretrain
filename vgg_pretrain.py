
import numpy as np
import torch
import math
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from vgg import *

# 1) Dataset:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
transform_test = transforms.Compose([transforms.ToTensor(), normalize])
cifar10_train = datasets.CIFAR10("data/cifar10", train=True, transform=transform_train, download=True)
print("dataload_cifar10_train_complete...")
cifar10_val = datasets.CIFAR10("data/cifar10", train=False, transform=transform_test, download=True)
print("dataload_cifar_val_complete...")

def validate():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 100 == 0 and i != 0:
                print("steps:", i, ", accuracy:", correct / total)

    print('Test accuracy: %f%%' % (
        100 * correct / total))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = vgg16()
model = vgg16(pretrain=False) # training from scratch.

# model = nn.DataParallel(model)
model.to(device)

batch_size = 512
data_loader = torch.utils.data.DataLoader(cifar10_val,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)
validate()        

# 12pm dec21
train_loader = torch.utils.data.DataLoader(cifar10_train,
                                          batch_size=256,
                                          shuffle=True,
                                          num_workers=8)
num_epochs = 200
import torch.optim as optim
# import pytorch_warmup as warmup

criterion = nn.CrossEntropyLoss()

model.train_fz_bn()

params_all_wo_bn = []
params_weight_bias = []
params_auxweight = []
names_weight = []
names_auxweight = []
layers = (nn.Conv2d, Conv2DLayerW, Conv2DLayerN, nn.BatchNorm2d, nn.Linear, 
          LinearLayerW, LinearLayerN, nn.ReLU, nn.Dropout, nn.MaxPool2d)

for name, param in model.named_parameters():
    params_weight_bias.append(param)
    if "weight" in name and int(name.split(".")[1]) not in [1,4,8,11,15,18,21,25,28,31,35,38,41]:
        names_weight.append(name)

optimizer_wb = torch.optim.AdamW(params_weight_bias, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)

num_steps = len(train_loader) * num_epochs
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
# warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

lambda_l1 = 1e-6
lambda_l2 = 5e-4

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_steps = 0
    running_correct = 0
    running_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # forward + backward + optimize
        optimizer_wb.zero_grad()
        outputs = model(inputs)
        ce_loss = criterion(outputs, labels)
        loss = ce_loss
        loss.backward()
        optimizer_wb.step()
        
        # compute statistics
        _, predicted = torch.max(outputs.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        running_steps += 1

        # print statistics
    print('Epoch %d, loss: %.3f' %
          (epoch + 1, running_loss / running_steps), "running_accuarcy:", running_correct / running_total)
        
print('Finished Training')

validate()