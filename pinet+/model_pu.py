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
        #
        # self.bn1 = nn.InstanceNorm1d(64)
        # self.bn2 = nn.InstanceNorm1d(128)
        # self.bn3 = nn.InstanceNorm1d(1024)
        # self.bn4 = nn.InstanceNorm1d(512)
        # self.bn5 = nn.InstanceNorm1d(256)


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
        # print(x.size())

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        # print(iden)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64,a='b'):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        
        if a=='b':
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        #
        # self.bn1 = nn.InstanceNorm1d(64)
        # self.bn2 = nn.InstanceNorm1d(128)
        # self.bn3 = nn.InstanceNorm1d(1024)
        # self.bn4 = nn.InstanceNorm1d(512)
        # self.bn5 = nn.InstanceNorm1d(256)

        self.k = k
        self.a = a

    def forward(self, x):
        batchsize = x.size()[0]
        if self.a=='b':
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.bn3(F.relu(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = self.bn4(F.relu(self.fc1(x)))
            x = self.bn5(F.relu(self.fc2(x)))
            x = self.fc3(x)
        elif self.a=='l':
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            
            x=F.relu(self.fc1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x =F.relu(self.fc2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
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
            # return torch.cat([x, pointfeat], 1), trans, trans_feat
            return x, pointfeat, trans, trans_feat

class PointNetfeat4(nn.Module):
    def __init__(self, d=5,global_feat = True, feature_transform = False):
        super(PointNetfeat4, self).__init__()
        # self.stn = STN3d()
        self.stn=STNkd(k=d)
        # self.conv1 = torch.nn.Conv1d(3, 64, 1)
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
            # return torch.cat([x, pointfeat], 1), trans, trans_feat
            return x, pointfeat, trans, trans_feat
        
        
class PointNetfeatRes(nn.Module):
    def __init__(self, d=5,global_feat = True, feature_transform = False,a='b'):
        super(PointNetfeatRes, self).__init__()
        # self.stn = STN3d()
        self.a=a
        self.stn=STNkd(k=d,a=a)
        # self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv11 = torch.nn.Conv1d(64, 64, 1)
        self.conv12 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv21 = torch.nn.Conv1d(128, 128, 1)
        self.conv22 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv31 = torch.nn.Conv1d(1024, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 1024, 1)
        if a=='b':
            self.bn1 = nn.BatchNorm1d(64)
            self.bn11 = nn.BatchNorm1d(64)
            self.bn12 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn21 = nn.BatchNorm1d(128)
            self.bn22 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn31 = nn.BatchNorm1d(256)
            self.bn32 = nn.BatchNorm1d(1024)
        
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
        if self.a=='b':
            x = self.bn1(F.relu(self.conv1(x)))
            residue=x
            x = self.bn11(F.relu(self.conv11(x)))
            x = self.bn12(F.relu(residue+self.conv12(x)))
#         x+=residue
        elif self.a=='l':
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            residue=x
            x = F.layer_norm(F.relu(self.conv11(x)),[x.size()[-1]])
            x = F.layer_norm(F.relu(residue+self.conv12(x)),[x.size()[-1]])
            
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        
        if self.a=='b':
            x = self.bn2(F.relu(self.conv2(x)))
            residue=x
            x = self.bn21(F.relu(self.conv21(x)))
            x = self.bn22(F.relu(residue+self.conv22(x)))
    #         x+=residue


            x = self.bn3(F.relu(self.conv3(x)))
            residue=x
            x = self.bn31(F.relu(self.conv31(x)))
            x = self.bn32(F.relu(residue+self.conv32(x)))
        elif self.a=='l':
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            residue=x
            x = F.layer_norm(F.relu(self.conv21(x)),[x.size()[-1]])
            x = F.layer_norm(F.relu(residue+self.conv22(x)),[x.size()[-1]])
    #         x+=residue

            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            residue=x
            x=F.relu(self.conv31(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(residue+self.conv32(x))
            x = F.layer_norm(x,[x.size()[-1]])
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            x = x.view(-1, 1024, 1)
            return x, pointfeat, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # return torch.cat([x, pointfeat], 1), trans, trans_feat
            return x, pointfeat, trans, trans_feat

class PointNetfeatResNostn(nn.Module):
    def __init__(self, d=5,global_feat = True, feature_transform = False,a='b'):
        super(PointNetfeatResNostn, self).__init__()
        # self.stn = STN3d()
        self.a=a
#         self.stn=STNkd(k=d,a=a)
        # self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv1 = torch.nn.Conv1d(d, 64, 1)
        self.conv11 = torch.nn.Conv1d(64, 64, 1)
        self.conv12 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv21 = torch.nn.Conv1d(128, 128, 1)
        self.conv22 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv31 = torch.nn.Conv1d(1024, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 1024, 1)
        if a=='b':
            self.bn1 = nn.BatchNorm1d(64)
            self.bn11 = nn.BatchNorm1d(64)
            self.bn12 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn21 = nn.BatchNorm1d(128)
            self.bn22 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn31 = nn.BatchNorm1d(256)
            self.bn32 = nn.BatchNorm1d(1024)
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
        if self.a=='b':
            x = self.bn1(F.relu(self.conv1(x)))
            residue=x
            x = self.bn11(F.relu(self.conv11(x)))
            x = self.bn12(F.relu(residue+self.conv12(x)))
#         x+=residue
        elif self.a=='l':
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            residue=x
            x = F.layer_norm(F.relu(self.conv11(x)),[x.size()[-1]])
            x = F.layer_norm(F.relu(residue+self.conv12(x)),[x.size()[-1]])
            
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        
        if self.a=='b':
            x = self.bn2(F.relu(self.conv2(x)))
            residue=x
            x = self.bn21(F.relu(self.conv21(x)))
            x = self.bn22(F.relu(residue+self.conv22(x)))
    #         x+=residue


            x = self.bn3(F.relu(self.conv3(x)))
            residue=x
            x = self.bn31(F.relu(self.conv31(x)))
            x = self.bn32(F.relu(residue+self.conv32(x)))
        elif self.a=='l':
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            residue=x
            x = F.layer_norm(F.relu(self.conv21(x)),[x.size()[-1]])
            x = F.layer_norm(F.relu(residue+self.conv22(x)),[x.size()[-1]])
    #         x+=residue

            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            residue=x
            x=F.relu(self.conv31(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(residue+self.conv32(x))
            x = F.layer_norm(x,[x.size()[-1]])
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            x = x.view(-1, 1024, 1)
            return x, pointfeat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # return torch.cat([x, pointfeat], 1), trans, trans_feat
            return x, pointfeat
        
class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        # self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        # self.conv4 = torch.nn.Conv1d(128, 1, 1)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        # self.ls=nn.LogSigmoid()
        # self.ls = torch.sigmoid()

    def forward(self, x1,x2):
        # batchsize = x.size()[0]
        batchsize = x1.size()[0]
        # n_pts = x.size()[2]
        n_pts = x1.size()[2]
        x1gf,x1pf, trans1, trans_feat1 = self.feat(x1)
        x2gf,x2pf, trans2, trans_feat2 = self.feat(x2)
        xf1=torch.cat([x1gf, x2gf], 1)
        xf2 = torch.cat([x2gf, x1gf], 1)
        # xf = xf.view(-1, 1024, 1).repeat(1, 1, n_pts)
        # print(x1.size())
        # print(xf.size())
        x1a=torch.cat([x1pf, xf1], 1)
        x2a = torch.cat([x2pf, xf2], 1)
        x=torch.cat((x1a,x2a),0)
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts*2, self.k)
        # self.k=1
        # x=x.view(-1, 1)
        # x = nn.LogSigmoid(x)
        # x = self.ls(x.view(-1, 1))
        # x=torch.sigmoid(x.view(-1, 1))
        # x = x.view(batchsize, n_pts * 2, 1)
        # return x, torch.cat((trans1,trans2),0), torch.cat((trans_feat1,trans_feat2),0)
        return x,trans1, trans_feat1

class PointNetDenseCls12(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5):
        super(PointNetDenseCls12, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        # self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeat4(global_feat=True, feature_transform=feature_transform,d=id)
        # self.minifeat=MiniPointNetfeat0(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        # self.conv0 = torch.nn.Conv1d(2112+id-3, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128+id-3, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=pdrop)
        # self.ls=nn.LogSigmoid()
        # self.ls = torch.sigmoid()

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
    
class PointNetDenseClsRes(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5,a='b'):
        super(PointNetDenseClsRes, self).__init__()
        self.k = k
        self.a = a
        self.feature_transform=feature_transform
        # self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeatRes(global_feat=True, feature_transform=feature_transform,d=id,a=a)
        # self.minifeat=MiniPointNetfeat0(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        # self.conv0 = torch.nn.Conv1d(2112+id-3, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128+id-3, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        if a=='b':
            self.bn0 = nn.BatchNorm1d(1024)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=pdrop)
        # self.ls=nn.LogSigmoid()
        # self.ls = torch.sigmoid()

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
        
        if self.a=='b':
            x = self.bn0(F.relu(self.dropout(self.conv0(x))))
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.bn3(F.relu(self.conv3(x)))

            x = self.bn4(F.relu(self.conv4(x)))
            
        elif self.a=='l':
            x=F.relu(self.dropout(self.conv0(x)))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv4(x))
            x = F.layer_norm(x,[x.size()[-1]])
            
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()

        x=x.view(-1, 1)

        x = x.view(batchsize, x1.size()[2]+x2.size()[2], 1)
        return x, trans_feat1, trans_feat2
    
class PointNetDenseClsBox(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5,a='b'):
        super(PointNetDenseClsRes, self).__init__()
        self.k = k
        self.a = a
        self.feature_transform=feature_transform
        # self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeatRes(global_feat=True, feature_transform=feature_transform,d=id,a=a)
        # self.minifeat=MiniPointNetfeat0(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        # self.conv0 = torch.nn.Conv1d(2112+id-3, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128+id-3, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        if a=='b':
            self.bn0 = nn.BatchNorm1d(1024)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=pdrop)
        # self.ls=nn.LogSigmoid()
        # self.ls = torch.sigmoid()

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
        
        if self.a=='b':
            x = self.bn0(F.relu(self.dropout(self.conv0(x))))
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.bn3(F.relu(self.conv3(x)))

            x = self.bn4(F.relu(self.conv4(x)))
            
        elif self.a=='l':
            x=F.relu(self.dropout(self.conv0(x)))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv4(x))
            x = F.layer_norm(x,[x.size()[-1]])
            
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()

        x=x.view(-1, 1)

        x = x.view(batchsize, x1.size()[2]+x2.size()[2], 1)
        return x, trans_feat1, trans_feat2
    
    
class PointNetDenseClsResScore(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5,a='b'):
        super(PointNetDenseClsResScore, self).__init__()
        self.k = k
        self.a = a
        self.feature_transform=feature_transform
        # self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeatRes(global_feat=True, feature_transform=feature_transform,d=id,a=a)
        # self.minifeat=MiniPointNetfeat0(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        # self.conv0 = torch.nn.Conv1d(2112+id-3, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128+id-3, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        if a=='b':
            self.bn0 = nn.BatchNorm1d(1024)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(64)
            
            self.bnpj0 = nn.BatchNorm1d(512)
            self.bnpj1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=pdrop)
        
        self.pj0=torch.nn.Linear(2048,512)
        self.pj1=torch.nn.Linear(512,128)
        self.pj2=torch.nn.Linear(128,1)
        # self.ls=nn.LogSigmoid()
        # self.ls = torch.sigmoid()

    def forward(self, x1,x2):
        batchsize = x1.size()[0]
        n_pts = x1.size()[2]
        x1gf,x1pf, trans1, trans_feat1 = self.feat(x1)
        x2gf,x2pf, trans2, trans_feat2 = self.feat(x2)
        
        px=torch.cat([x1gf, x2gf], 1).squeeze()
        if self.a=='b':
            px = self.bnpj0(F.relu(self.dropout(self.pj0(px))))
            px = self.bnpj1(F.relu(self.pj1(px)))
            px = torch.sigmoid(self.pj2(px))

        xf1=torch.cat([x1gf, x2gf], 1)
        xf2 = torch.cat([x2gf, x1gf], 1)

        xf1=xf1.repeat(1,1,x1pf.size()[2])
        xf2 = xf2.repeat(1, 1, x2pf.size()[2])
        x1a=torch.cat([x1pf, xf1], 1)
        x2a = torch.cat([x2pf, xf2], 1)
        x=torch.cat((x1a,x2a),2)
        
        if self.a=='b':
            x = self.bn0(F.relu(self.dropout(self.conv0(x))))
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.bn3(F.relu(self.conv3(x)))

            x = self.bn4(F.relu(self.conv4(x)))
            
        elif self.a=='l':
            x=F.relu(self.dropout(self.conv0(x)))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv4(x))
            x = F.layer_norm(x,[x.size()[-1]])
            
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()

        x=x.view(-1, 1)

        x = x.view(batchsize, x1.size()[2]+x2.size()[2], 1)
        return x, px
    
class PointNetDenseClsResScoreStn(nn.Module):
    def __init__(self, k = 2, feature_transform=False,pdrop=0.0,id=5,a='b'):
        super(PointNetDenseClsResScoreStn, self).__init__()
        self.k = k
        self.a = a
        self.stn_c=STNkd(k=id,a=a)
        self.stn_m=STNkd(k=id,a=a)
        self.feature_transform=feature_transform
        # self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeatResNostn(global_feat=True, feature_transform=feature_transform,d=id,a=a)
        # self.minifeat=MiniPointNetfeat0(global_feat=True, feature_transform=feature_transform,d=id)
        self.conv0 = torch.nn.Conv1d(2112, 1024, 1)
        # self.conv0 = torch.nn.Conv1d(2112+id-3, 1024, 1)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128+id-3, 64, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        if a=='b':
            self.bn0 = nn.BatchNorm1d(1024)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(64)
            
            self.bnpj0 = nn.BatchNorm1d(512)
            self.bnpj1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=pdrop)
        
        self.pj0=torch.nn.Linear(2048,512)
        self.pj1=torch.nn.Linear(512,128)
        self.pj2=torch.nn.Linear(128,1)
        # self.ls=nn.LogSigmoid()
        # self.ls = torch.sigmoid()

    def forward(self, x1,x2):
        batchsize = x1.size()[0]
        n_pts = x1.size()[2]
        
        trans1 = self.stn_c(x1)
        x1 = x1.transpose(2, 1)
        x1 = torch.bmm(x1, trans1)
        x1 = x1.transpose(2, 1)
        
        trans2 = self.stn_c(x2)
        x2 = x2.transpose(2, 1)
        x2 = torch.bmm(x2, trans2)
        x2 = x2.transpose(2, 1)
        
        trans2m = self.stn_m(torch.cat((x1,x2),2))
        x2 = x2.transpose(2, 1)
        x2 = torch.bmm(x2, trans2m)
        x2 = x2.transpose(2, 1)
        
        x1gf,x1pf = self.feat(x1)
        x2gf,x2pf = self.feat(x2)
        
        px=torch.cat([x1gf, x2gf], 1).squeeze()
        if self.a=='b':
            px = self.bnpj0(F.relu(self.dropout(self.pj0(px))))
            px = self.bnpj1(F.relu(self.pj1(px)))
            px = torch.sigmoid(self.pj2(px))

        xf1=torch.cat([x1gf, x2gf], 1)
        xf2 = torch.cat([x2gf, x1gf], 1)

        xf1=xf1.repeat(1,1,x1pf.size()[2])
        xf2 = xf2.repeat(1, 1, x2pf.size()[2])
        x1a=torch.cat([x1pf, xf1], 1)
        x2a = torch.cat([x2pf, xf2], 1)
        x=torch.cat((x1a,x2a),2)
        
        if self.a=='b':
            x = self.bn0(F.relu(self.dropout(self.conv0(x))))
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.bn3(F.relu(self.conv3(x)))

            x = self.bn4(F.relu(self.conv4(x)))
            
        elif self.a=='l':
            x=F.relu(self.dropout(self.conv0(x)))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv1(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv2(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv3(x))
            x = F.layer_norm(x,[x.size()[-1]])
            x=F.relu(self.conv4(x))
            x = F.layer_norm(x,[x.size()[-1]])
            
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()

        x=x.view(-1, 1)

        x = x.view(batchsize, x1.size()[2]+x2.size()[2], 1)
        return x, px

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
