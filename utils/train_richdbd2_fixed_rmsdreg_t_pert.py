from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset3rnt
# from pointnet.model import PointNetDenseCls7, feature_transform_regularizer
# from 16
# from pointnet.dataset import ShapeNetDataset5
from pointnet.model import PointNetDenseCls12lregtrans, feature_transform_regularizer

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score,recall_score,roc_curve,auc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help="start epoch")

parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--npoints', type=int, default=20000, help='subsample points')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--class_choice', type=str, default='protein', help="class_choice")
parser.add_argument('--r', type=str, default='recept', help="recept_choice")
parser.add_argument('--l', type=str, default='ligand', help="ligand_choice")
parser.add_argument('--bs2', type=int, default=8, help="bs")
parser.add_argument('--drop', type=int, default=0, help="droprate")
parser.add_argument('--ft', type=int, default=0, help="ft")
# from 16
parser.add_argument('--indim', type=int, default=5, help="input dim")
parser.add_argument('--randdiam', type=int, default=5, help="rand dim")
parser.add_argument('--start', type=int, default=10, help="start epoch")
parser.add_argument('--fac', type=int, default=100, help="start epoch")
parser.add_argument('--lloss', type=int, default=1, help="start epoch")
parser.add_argument('--rs', type=int, default=0, help="start epoch")
parser.add_argument('--rss', type=int, default=3000, help="start epoch")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
if opt.ft==1:
    opt.feature_transform=True
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset_r = ShapeNetDataset3rnt(
    root=opt.dataset,
    npoints=opt.rss,
    classification=False,
    class_choice=[opt.r],
    indim=opt.indim,
    rs=opt.rs)
dataloader_r = torch.utils.data.DataLoader(
    dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

dataset_l = ShapeNetDataset3rnt(
    root=opt.dataset,
    npoints=opt.rss,
    classification=False,
    class_choice=[opt.l],
    indim=opt.indim,
    rs=opt.rs)
dataloader_l = torch.utils.data.DataLoader(
    dataset_l,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_r = ShapeNetDataset3rnt(
    root=opt.dataset,
    npoints=opt.rss,
    classification=False,
    class_choice=[opt.r],
    split='test',
    data_augmentation=False,
    indim=opt.indim,
    rs=opt.rs)
testdataloader_r = torch.utils.data.DataLoader(
    test_dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_l = ShapeNetDataset3rnt(
    root=opt.dataset,
    npoints=opt.rss,
    classification=False,
    class_choice=[opt.l],
    split='test',
    data_augmentation=False,
    indim=opt.indim,
    rs=opt.rs)
testdataloader_l = torch.utils.data.DataLoader(
    test_dataset_l,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(dataset_r), len(test_dataset_r))
num_classes = dataset_r.num_seg_classes
print('classes', num_classes)

print(len(dataset_l), len(test_dataset_l))
num_classes = dataset_l.num_seg_classes
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'
# print(opt.feature_transform)
# classifier = PointNetDenseCls7(k=num_classes, feature_transform=opt.feature_transform,pdrop=1.0*opt.drop/10.0)
classifier = PointNetDenseCls12lregtrans(k=num_classes, feature_transform=opt.feature_transform,pdrop=1.0*opt.drop/10.0,id=opt.indim)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

# loss=BCEloss

num_batch = len(dataset_r) / opt.batchSize
# randpert_train=torch.rand((len(dataset_r),11))*10-5
# randpert_test=torch.rand((len(test_dataset_r),11))*10-5
randpert_train = torch.rand((len(dataset_r), 11)) * 2*opt.randdiam - opt.randdiam
randpert_test = torch.rand((len(test_dataset_r), 11)) * 2*opt.randdiam - opt.randdiam

for epoch in range(opt.nepoch):
    # randpert_train = torch.rand((len(dataset_r), 11)) * 2*opt.randdiam - opt.randdiam
    # randpert_test = torch.rand((len(test_dataset_r), 11)) * 2*opt.randdiam - opt.randdiam
    scheduler.step()
    optimizer.zero_grad()
    show_flag = 0
    totalloss = 0
    print(epoch)
    for i, (datar,datal) in enumerate(zip(dataloader_r,dataloader_l), 0):
        pointsr, targetr = datar
        pointsl,targetl=datal

        if targetr.sum() == 0 or targetl.sum() == 0:
            continue

        memlim=opt.npoints
        # sligand=pointsl.size()[1]
        # srecep=pointsr.size()[1]
        if pointsl.size()[1]+pointsr.size()[1]>memlim:
            lr=pointsl.size()[1]*memlim/(pointsl.size()[1]+pointsr.size()[1])
            rr = pointsr.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
            ls=np.random.choice(pointsl.size()[1], lr, replace=False)
            rs=np.random.choice(pointsr.size()[1], rr, replace=False)
            pointsr=pointsr[:,rs,:]
            targetr=targetr[:,rs]
            pointsl=pointsl[:,ls,:]
            targetl=targetl[:,ls]

        pointsr = pointsr.transpose(2, 1)
        pointsl = pointsl.transpose(2, 1)
        # reg=torch.mean(pointsl,dim=2)
        # reg[:,3:]=0
        # reg=reg.cuda()
        # cr=torch.mean(pointsr,dim=2)
        # cr[:, 3:] = 0
        # cl = torch.mean(pointsl, dim=2)
        # cl[:, 3:] = 0
        # reg=cl-cr
        # cr=cr.cuda()
        # cl=cl.cuda()

        reg=randpert_train[i,:].view(1,-1)
        reg[:, 3:] = 0
        reg=reg.cuda()
        pointsr, targetr = pointsr.cuda(), targetr.cuda()
        pointsl, targetl = pointsl.cuda(), targetl.cuda()


        # classifier = classifier.train()
        classifier = classifier.eval()
        for m in classifier.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        # regpred = classifier(pointsr-cr.unsqueeze(2),pointsl-cl.unsqueeze(2))
        regpred = classifier(pointsr,pointsl-reg.unsqueeze(2))


        regpred=regpred.view(1,-1)
        loss=1.0/opt.bs2*F.mse_loss(regpred,reg[:,0:3].float())
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat1) * 0.001/opt.bs2
            loss += feature_transform_regularizer(trans_feat2) * 0.001 / opt.bs2
        totalloss+=loss.item()
        loss.backward()
        show_flag = 0
        if (i + 1) % opt.bs2 == 0:
            optimizer.step()
            optimizer.zero_grad()
            # totalloss=0
            show_flag = 1

        if i == len(dataset_l) - 1:
            optimizer.step()
            optimizer.zero_grad()
            # totalloss = 0
            show_flag = 1

        if show_flag:

            print('[%d: %d/%d] train loss: %f ' % (epoch, i, num_batch, totalloss))


            totalloss = 0


        # if i % 4000 == 0:
        if i ==len(dataset_r)-1:
            testloss=0
            for j, (datar, datal) in enumerate(zip(testdataloader_r, testdataloader_l), 0):
                # pointsr, targetr = datar
                # pointsl, targetl = datal
                pointsr, targetr= datar
                pointsl, targetl= datal
                # reglabel=torch.cat([tr-tl,ror,rol])
                # ror = ror.view(1, 1)
                # rol = rol.view(1, 1)
                # print(tr.size())
                # print(ror.size())
                # reglabel = torch.cat([tr - tl, ror, rol], 1)
                # print(pointsr.size())
                # memlim = 110000
                memlim = 90000
                if pointsl.size()[1] + pointsr.size()[1] > memlim:
                    # print(pointsl.size()[1] + pointsr.size()[1])
                    lr = pointsl.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
                    rr = pointsr.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
                    ls = np.random.choice(pointsl.size()[1], lr, replace=False)
                    rs = np.random.choice(pointsr.size()[1], rr, replace=False)

                    pointsr = pointsr[:, rs, :]
                    targetr = targetr[:, rs]
                    pointsl = pointsl[:, ls, :]
                    targetl = targetl[:, ls]
                pointsr = pointsr.transpose(2, 1)
                pointsl = pointsl.transpose(2, 1)
                # reg = torch.mean(pointsl, dim=2)
                # reg[:, 3:] = 0
                # reg = reg.cuda()
                # cr = torch.mean(pointsr, dim=2)
                # cr[:, 3:] = 0
                # cl = torch.mean(pointsl, dim=2)
                # cl[:, 3:] = 0
                # reg = cl - cr
                # cr = cr.cuda()
                # cl = cl.cuda()
                reg = randpert_test[j, :].view(1,-1)
                reg[:, 3:] = 0
                reg = reg.cuda()
                pointsr, targetr = pointsr.cuda(), targetr.cuda()
                pointsl, targetl = pointsl.cuda(), targetl.cuda()
                classifier = classifier.eval()
                try:
                    # predreg = classifier(pointsr-cr.unsqueeze(2), pointsl-cl.unsqueeze(2))
                    predreg = classifier(pointsr, pointsl-reg.unsqueeze(2))
                except:
                    continue

                testloss+=F.mse_loss(predreg,reg[:,0:3].float()).item()
            print('testloss:')
            print(testloss)
            # testloss = 0


    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
