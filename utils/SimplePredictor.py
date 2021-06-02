from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pinet.model import PointNetDenseCls12, feature_transform_regularizer
import sys
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


random.seed(random.randint(1, 10000) )
torch.manual_seed(random.randint(1, 10000) )

filel=sys.argv[1]
filer=sys.argv[2]


num_classes = 2
classifier = PointNetDenseCls12(k=num_classes, feature_transform=False,pdrop=0.0,id=5)

classifier.cuda()


PATH='../models/dbd_aug.pth'
classifier.load_state_dict(torch.load(PATH))
classifier.eval()

pointsr=np.loadtxt(filer).astype(np.float32)
pointsl=np.loadtxt(filel).astype(np.float32)

coordsetr = pointsr[:, 0:3]
featsetr = pointsr[:, 3:]

coordsetl = pointsl[:, 0:3]
featsetl = pointsl[:, 3:]

featsetr = featsetr / np.sqrt(np.max(featsetr ** 2, axis=0))
featsetl = featsetl / np.sqrt(np.max(featsetl ** 2, axis=0))
        
coordsetr = coordsetr - np.expand_dims(np.mean(coordsetr, axis=0), 0)  # center
coordsetl = coordsetl - np.expand_dims(np.mean(coordsetl, axis=0), 0)  # center

pointsr[:, 0:5] = np.concatenate((coordsetr, featsetr), axis=1)
pointsl[:, 0:5] = np.concatenate((coordsetl, featsetl), axis=1)

pointsr=torch.from_numpy(pointsr).unsqueeze(0)
pointsl=torch.from_numpy(pointsl).unsqueeze(0)


memlim=120000
if pointsl.size()[1] + pointsr.size()[1] > memlim:
    lr = pointsl.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
    rr = pointsr.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
    ls = np.random.choice(pointsl.size()[1], lr, replace=False)
    rs = np.random.choice(pointsr.size()[1], rr, replace=False)

    pointsr = pointsr[:, rs, :]
    pointsl = pointsl[:, ls, :]
    
pointsr = pointsr.transpose(2, 1).cuda()
pointsl = pointsl.transpose(2, 1).cuda()

classifier = classifier.eval()

pred, _, _ = classifier(pointsr,pointsl)

pred = pred.view(-1, 1)

np.savetxt(filel[0:4]+'_prob_r_l.seg',torch.sigmoid(pred).view(1, -1).data.cpu())
    
