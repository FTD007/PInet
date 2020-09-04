from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parameter
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset3aug
from pointnet.model import PointNetDenseCls12, feature_transform_regularizer

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score,recall_score,roc_curve,auc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def gk(x):
    cen=torch.nn.Parameter(torch.tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]).float(), requires_grad=False).cuda()
    xmat=x.float().cuda()-cen
    sigma=200
    y = torch.sum(torch.sigmoid(sigma * (xmat + 0.1 / 2)) - torch.sigmoid(sigma * (xmat - 0.1 / 2)),dim=0)
    y = y / torch.sum(y)
    return y

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--npoints', type=int, default=20000, help='subsample points')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--class_choice', type=str, default='protein', help="class_choice")
parser.add_argument('--r', type=str, default='recept', help="recept_choice")
parser.add_argument('--l', type=str, default='ligand', help="ligand_choice")
parser.add_argument('--fold', type=str, default='', help="kfold")
parser.add_argument('--bs2', type=int, default=8, help="bs")
parser.add_argument('--drop', type=int, default=0, help="droprate")
parser.add_argument('--ft', type=int, default=0, help="ft")
# from 16
parser.add_argument('--indim', type=int, default=5, help="input dim")
parser.add_argument('--start', type=int, default=10, help="start epoch")
parser.add_argument('--fac', type=float, default=100, help="start epoch")
parser.add_argument('--lloss', type=int, default=1, help="start epoch")
parser.add_argument('--rs', type=int, default=0, help="start epoch")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
if opt.ft==1:
    opt.feature_transform=True
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# data loading
dataset_r = ShapeNetDataset3aug(
    root=opt.dataset,
    npoints=3000,
    classification=False,
    class_choice=[opt.r],
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
dataloader_r = torch.utils.data.DataLoader(
    dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

dataset_l = ShapeNetDataset3aug(
    root=opt.dataset,
    npoints=3000,
    classification=False,
    class_choice=[opt.l],
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
dataloader_l = torch.utils.data.DataLoader(
    dataset_l,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_r = ShapeNetDataset3aug(
    root=opt.dataset,
    npoints=3000,
    classification=False,
    class_choice=[opt.r],
    split='test',
    data_augmentation=False,
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
testdataloader_r = torch.utils.data.DataLoader(
    test_dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_l = ShapeNetDataset3aug(
    root=opt.dataset,
    npoints=3000,
    classification=False,
    class_choice=[opt.l],
    split='test',
    data_augmentation=False,
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
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
# model
classifier = PointNetDenseCls12(k=num_classes, feature_transform=opt.feature_transform,pdrop=1.0*opt.drop/10.0,id=opt.indim)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()


num_batch = len(dataset_r) / opt.batchSize

for epoch in range(opt.nepoch):
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
        # subsample
        memlim=opt.npoints
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
        pointsr, targetr = pointsr.cuda(), targetr.cuda()
        pointsl, targetl = pointsl.cuda(), targetl.cuda()

        classifier = classifier.eval()
        for m in classifier.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        pred, trans_feat1,trans_feat2 = classifier(pointsr,pointsl)
        pred = pred.view(-1, 1)
        target=torch.cat((targetr,targetl),1)
        target = target.view(-1, 1) - 1
        loss = 1.0/opt.bs2*F.binary_cross_entropy_with_logits(pred, target.float(),pos_weight=torch.FloatTensor([(target.size()[0]-float(target.cpu().sum()))*1.0/float(target.cpu().sum())]).cuda())
        # manual batch loss aggregate
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat1) * 0.001/opt.bs2
            loss += feature_transform_regularizer(trans_feat2) * 0.001 / opt.bs2
        if epoch>=opt.start:
            # compute hist loss
            predr=pred[0:targetr.size()[1]].view(-1)
            predl=pred[targetr.size()[1]:].view(-1)
            br=(torch.gt(torch.sigmoid(predr.data),0.5)==1).nonzero()
            bl = (torch.gt(torch.sigmoid(predl.data), 0.5) == 1).nonzero()
            if br.size()[0]!=0 and bl.size()[0]!=0:
                if br.size()[0] > 5000:

                    brs = np.random.choice(br.size()[0], 5000, replace=False)
                    pr=torch.transpose(pointsr[0,0:3,br[brs]].view(3,-1),0,1)
                else:
                    pr = torch.transpose(pointsr[0, 0:3, br].view(3, -1),0,1)

                if bl.size()[0] > 5000:

                    bls = np.random.choice(bl.size()[0], 5000, replace=False)
                    pl=torch.transpose(pointsl[0,0:3,bl[bls]].view(3,-1),0,1)
                else:
                    pl = torch.transpose(pointsl[0, 0:3, bl].view(3, -1), 0, 1)
                dr=pairwise_distances(pr)
                dl=pairwise_distances(pl)
                diam=torch.max(torch.tensor([torch.max(dr),torch.max(dl)]))
                dr=dr/diam
                dl = dl / diam

                if opt.lloss==1:
                    loss+=opt.fac*F.l1_loss(gk(dr.view(-1,1)),gk(dl.view(-1,1)))
                else:
                    loss+=opt.fac*F.mse_loss(gk(dr.view(-1,1)),gk(dl.view(-1,1)))

        totalloss+=loss
        loss.backward()
        show_flag = 0
        if (i + 1) % opt.bs2 == 0:
            optimizer.step()
            optimizer.zero_grad()
            show_flag = 1

        if i == len(dataset_l) - 1:
            optimizer.step()
            optimizer.zero_grad()
            show_flag = 1

        if show_flag:
            pred_choice = torch.gt(torch.sigmoid(pred.data),0.5).long()
            correct0 = pred_choice.eq(target.data)
            correct0 = pred_choice.eq(target.data).cpu().sum()
            correct1 = (pred_choice.eq(target.data).long()*target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, totalloss.item(), correct0.item() / float(opt.batchSize * target.size()[0])))
            print('[%d: %d/%d] train loss: %f class 1 accuracy: %f' % (
                epoch, i, num_batch, totalloss.item(), correct1.item() / float(target.data.sum().item())))

            totalloss = 0


        if i % 8000 == 0:
            all = []
            allp = []
            allr = []
            allauc = []
            for j, (datar, datal) in enumerate(zip(testdataloader_r, testdataloader_l), 0):
                pointsr, targetr = datar
                pointsl, targetl = datal

                memlim = 90000
                if pointsl.size()[1] + pointsr.size()[1] > memlim:
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

                pointsr, targetr = pointsr.cuda(), targetr.cuda()
                pointsl, targetl = pointsl.cuda(), targetl.cuda()
                classifier = classifier.eval()
                try:
                    pred, _, _ = classifier(pointsr, pointsl)
                except:
                    continue
                pred = pred.view(-1, 1)
                target = torch.cat((targetr, targetl), 1)
                target = target.view(-1, 1) - 1

                pred_choice = torch.gt(torch.sigmoid(pred.data), 0.5).long()

                correct0 = pred_choice.eq(target.data).cpu().sum()
                correct1 = (pred_choice.eq(target.data).long() * target.data).cpu().sum()

                blue = lambda x: '\033[94m' + x + '\033[0m'

                if j==0:
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(),
                                                                    correct0.item() / float(
                                                                        opt.batchSize * target.size()[0])))
                    print('[%d: %d/%d] %s loss: %f class 1 accuracy: %f' % (
                        epoch, i, num_batch, blue('test'), loss.item(),
                        correct1.item() / float(target.data.sum().item())))

                all.append(correct0.item() / float(opt.batchSize * target.size()[0]))

                allp.append(precision_score(target.data.cpu(), pred_choice.cpu()))

                allr.append(recall_score(target.data.cpu(), pred_choice.cpu()))
                fpr, tpr, thresholds = roc_curve(target.data.cpu(),
                                                 torch.sigmoid(pred.data).cpu(), pos_label=1)

                allauc.append(auc(fpr, tpr))
            print(sum(all) * 1.0 / len(all))
            print(sum(allp) * 1.0 / len(all))
            print(sum(allr) * 1.0 / len(all))
            print(sum(allauc) * 1.0 / len(all))

    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
