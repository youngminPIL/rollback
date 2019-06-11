# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net
#from tensorboard_logger import configure, log_value
import json
import network as nw
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir',default='/home/zzd/Market/pytorch',type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
opt = parser.parse_args()

#torch.manual_seed(7)

data_dir = opt.data_dir
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

name = 'ft_ResNet50'
data_dir = '/home/ro/Reid/Market/pytorch'
#num_class = 751

dir_name = 'experiment_Result'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name = 'experiment_Result/p4'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

#configure(dir_name)
print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
gpu_ids[0] = 0
print(gpu_ids[0])
opt.batchsize = 32
resize = (288,144)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])


######################################################################
# Load Data
# ---------

transform_train_list = [
        transforms.Resize(resize, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(resize,interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]



print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all' ),
                                          data_transforms['train'])


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=16)
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))

######################################################################
# Training the model
# ------------------

y_loss = {} # loss history
y_loss['train'] = []

y_err = {}
y_err['train'] = []

def get_gradsum(temp):
    a, b, c, d = temp.size()
    ans = torch.sum(temp.pow(2)) / (a * b * c * d)
    return ans
def get_gradsum2(temp):
    a, b = temp.size()
    ans = torch.sum(temp.pow(2)) / (a * b)
    return ans


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
         
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            running_corrects = running_corrects.float()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                phase, epoch_loss, epoch_acc))

            #if phase == 'train':
            #    log_value('train_loss', epoch_loss, epoch)
            #    log_value('train_acc', epoch_acc, epoch)
                

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)             
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model



######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(dir_name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])
        #nn.DataParallel(network, device_ids=[2,3]).cuda()

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network



model_structure = ft_net(len(class_names))
target = 'experiment_Result/p3'
path = target+ '/ft_ResNet50/net_39.pth'
print('origin_with: ' + target)
model = load_network_path(model_structure, path)


model_from = ft_net(len(class_names))

nw.dup_net6(model_from, model, 'model.conv1', 'model.bn1', 'model.layer1','model.layer2', 'model.fc', 'classifier') #C1L1L2+FC

if use_gpu:
    model = model.cuda()
    #nn.DataParallel(model, device_ids=[2,3]).cuda()
criterion = nn.CrossEntropyLoss()

ignored_params = list(map(id, model.model.layer2.parameters())) + list(map(id, model.model.layer3.parameters())) + \
                 list(map(id, model.model.layer4.parameters())) + list(map(id, model.model.fc.parameters())) \
                 + list(map(id, model.classifier.parameters())) + list(map(id, model.model.layer1.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
print(base_params)
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.001},  
    {'params': model.model.layer1.parameters(), 'lr': 0.001},
    {'params': model.model.layer2.parameters(), 'lr': 0.001},
    {'params': model.model.layer3.parameters(), 'lr': 0.01},
    {'params': model.model.layer4.parameters(), 'lr': 0.01},
    {'params': model.model.fc.parameters(), 'lr': 0.01},
    {'params': model.classifier.parameters(), 'lr': 0.01}
], momentum=0.9, weight_decay=5e-4, nesterov=True)



# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join(dir_name,name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=40)

