from __future__ import print_function
#from random import shuffle
#from torchvision.transforms import functional as F

#%matplotlib inline
import argparse
import random
import torch
import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
#import torchvision.datasets as dset
#import torchvision.utils as vutils
#import numpy as np

#safely create directory for output
import os
from datetime import datetime
output_dir = './output/filter_functions'+str(datetime.now())[:19]
models_dir = output_dir+'/models'
logs_dir = output_dir+'/logs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(models_dir)
    os.makedirs(logs_dir)
log_training = open(logs_dir+'/log_train','a')
log_validating = open(logs_dir+'/log_val','a')
log_training.close()
log_validating.close()

# self defined classes
from fcn_model import FCNs, VGGNet, weights_init_normal
from data_loader import get_loader

root_lab = './dataset/ts0126_80k/'
#root_unlab = PATH_TO_UNLABELED
cuda_device = "cuda:0"
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net


dataroot = root_lab
workers = 0
batch_size = 128
num_epochs = 50

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device(cuda_device if (torch.cuda.is_available() and ngpu > 0) else "cpu")

BATCH_SIZE = 1

train_loader, val_loader = get_loader(root_path=root_lab,image_size=256,batch_size=BATCH_SIZE,split_ratio=0.999)

label_iter = iter(train_loader)
val_iter = iter(val_loader)

# Initialize BCELoss function
#BceLoss = nn.BCELoss()
#CELoss = nn.CrossEntropyLoss()

#CELoss = nn.CrossEntropyLoss(weight=torch.FloatTensor([223,1]).to(device))
#MSELoss = nn.MSELoss()

num_train_iter = len(train_loader)
num_val_iter = len(val_loader)

loss_list = []

Val_interval = 1000
best_val = 99999
n_class=1
h = 256
vgg_model = VGGNet(requires_grad=True)

model = FCNs(pretrained_net=vgg_model, n_class=n_class, h = 256, device=device,
                singlelinear = True, singleconv = False)
model.apply(weights_init_normal)
model.to(device)

#model.load_state_dict(torch.load('models/fcn_baseline/epoch2iter9000val18.831.pth'))

optimizerAll = optim.Adam(model.parameters(), lr=0.001, betas=(beta1, 0.999))

MSELoss = nn.MSELoss()

#num_train_iter = len(train_loader)
loss_list = []

### TRAIN

for epoch in range(num_epochs):
    # For each batch in the dataloader
    label_iter = iter(train_loader)
    for iters in range(num_train_iter):
        model.zero_grad()
        #################################
        ########get labeled image:#######
        #################################
        img, vec, mat, path = label_iter.next()
        label = mat.type(torch.long)
        label.squeeze_(1)
        
        img_gpu = img.to(device)
        lab_gpu = label.to(device)
        vec_gpu = vec.to(device)

        output = model(img_gpu).squeeze_(-1)
        
        loss_vec = MSELoss(output, vec_gpu)
        loss_vec.backward(retain_graph=True)
        
        #loss_lab = CELoss(output, lab_gpu) #+ errD_lab
        #loss_lab.backward()
        
        #optimizerD.step()
        optimizerAll.step()
        print(epoch, iters, loss_vec.item())
        log_training = open(logs_dir+'/log_train','a')
        log_training.write('{}  {}  {} \n'.format(epoch, iters, loss_vec.item()))
        log_training.close()
        loss_list.append(loss_vec.item())

        if (iters % Val_interval == 999):
            #### VALIDATION
            with torch.no_grad():
                val_iter = iter(val_loader)
                model.eval()
                loss_sum = 0
                ###
                for val_iters in range(num_val_iter):
                    img, vec, mat, path = val_iter.next()
                    label = mat.type(torch.long)
                    label.squeeze_(1)                
                    img_gpu = img.to(device)
                    lab_gpu = label.to(device)
                    vec_gpu = vec.to(device)                
                    output = model(img_gpu).squeeze_(-1)                
                    loss_vec = MSELoss(output, vec_gpu)
                    #print(epoch, iters, loss_vec.item())

                    loss_sum += loss_vec.item()
                ###
                val_loss = loss_sum / num_val_iter
                log_validating = open(logs_dir+'/log_val','a')
                log_validating.write('{}  {}  {} \n'.format(epoch, iters, val_loss))
                log_validating.close()
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(),
                    models_dir + '/epoch{}iter{}val{:06.3f}.pth'.format(epoch,iters,val_loss))
                    #'./models/'+'epoch'+str(epoch)+'iter'+str(iters)+'val'+str(val_loss)+'.pth')
                model.train()
