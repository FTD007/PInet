from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
import math

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)



    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        # print(iden)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

# geo and physical chemical
class PointNetfeat4(nn.Module):
    def __init__(self, d=5,global_feat = True, feature_transform = False):
        super(PointNetfeat4, self).__init__()
        self.stn=STNkd(k=d)
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            x = x.view(-1, 1024, 1)
            return x, pointfeat, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return x, pointfeat, trans, trans_feat

#geometry only
class PointNetfeat4geo(nn.Module):
    def __init__(self, d=5,global_feat = True, feature_transform = False):
        super(PointNetfeat4geo, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        geox=x[:,0:3,:]
        fx=x[:,3:,:]
        trans = self.stn(geox)
        geox = geox.transpose(2, 1)
        geox = torch.bmm(geox, trans)
        geox = geox.transpose(2, 1)
        x=torch.cat((geox,fx),dim=1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            x = x.view(-1, 1024, 1)
            return x, pointfeat, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return x, pointfeat, trans, trans_feat

# general use one
class PointNetDenseCls12(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5):
        super(PointNetDenseCls12, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat4(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=pdrop)

    def forward(self, x1,x2):
        batchsize = x1.size()[0]
        n_pts = x1.size()[2]
        x1gf,x1pf, trans1, trans_feat1 = self.feat(x1)
        x2gf,x2pf, trans2, trans_feat2 = self.feat(x2)

        # global
        xf1=torch.cat([x1gf, x2gf], 1)
        xf2 = torch.cat([x2gf, x1gf], 1)

        # point feat concat with global
        xf1=xf1.repeat(1,1,x1pf.size()[2])
        xf2 = xf2.repeat(1, 1, x2pf.size()[2])
        x1a=torch.cat([x1pf, xf1], 1)
        x2a = torch.cat([x2pf, xf2], 1)
        x=torch.cat((x1a,x2a),2)

        # MLP
        x = F.relu(self.bn0(self.dropout(self.conv0(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()

        x=x.view(-1, 1)

        x = x.view(batchsize, x1.size()[2]+x2.size()[2], 1)
        return x, trans_feat1, trans_feat2

# geo only
class PointNetDenseCls12geo(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5):
        super(PointNetDenseCls12geo, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat4geo(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=pdrop)

    def forward(self, x1,x2):
        batchsize = x1.size()[0]
        n_pts = x1.size()[2]
        x1gf,x1pf, trans1, trans_feat1 = self.feat(x1)
        x2gf,x2pf, trans2, trans_feat2 = self.feat(x2)

        xf1=torch.cat([x1gf, x2gf], 1)
        xf2 = torch.cat([x2gf, x1gf], 1)

        xf1=xf1.repeat(1,1,x1pf.size()[2])
        xf2 = xf2.repeat(1, 1, x2pf.size()[2])
        x1a=torch.cat([x1pf, xf1], 1)
        x2a = torch.cat([x2pf, xf2], 1)
        x=torch.cat((x1a,x2a),2)
        x = F.relu(self.bn0(self.dropout(self.conv0(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()

        x=x.view(-1, 1)
        x = x.view(batchsize, x1.size()[2]+x2.size()[2], 1)

        return x, trans_feat1, trans_feat2

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
