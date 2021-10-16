import os
import json
import numpy as np

import torch
import torch.nn.functional as F
import pickle

from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import sys
from scipy.spatial import distance







class ProteinDatasetSam(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,normsize=0,subsample=True,mul=30,exppower=1,folder='dbdapbsfix',nop=2048,dcut=10,powerd=1,hf=0,kl=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
        self.filelist = json.load(open(file, 'r'))
        self.mul=mul
        self.folder=folder
        self.nop=nop
        self.pd=powerd
        self.dcut=dcut
        self.half=hf
        self.expp=exppower
        self.normsize=normsize
        self.maxradius=76
        self.pkl=kl
        


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
        plf=self.folder+'/pts_ne2/'+self.filelist[cindex][-6:-1]+'l.pts'
        prf=self.folder+'/pts_ne2/'+self.filelist[cindex][-6:-1]+'r.pts'
        
        llf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.seg'
        lrf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.seg'
        
        if self.pkl:
            dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.ssseg'
            drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.ssseg'
        else:
            dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.sseg'
            drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.sseg'
        
        tdplf=np.loadtxt(plf)
        tdprf=np.loadtxt(prf)
        
        if tdplf.shape[0]>tdprf.shape[0]:
            
            dplf=tdplf
            dprf=tdprf
        
            dllf=np.loadtxt(llf)
            dlrf=np.loadtxt(lrf)
            
            ddlf=np.loadtxt(dlf)
            ddrf=np.loadtxt(drf)
        else:
            dplf=tdprf
            dprf=tdplf
        
            dllf=np.loadtxt(lrf)
            dlrf=np.loadtxt(llf)
            
            ddlf=np.loadtxt(drf)
            ddrf=np.loadtxt(dlf)
            
        if self.pd==None:
            pass
        elif self.pd[0:3]=='exp':
            ddlf=np.exp(-(np.minimum(ddlf,self.dcut)/float(self.pd[3:]))**self.expp)
            ddrf=np.exp(-(np.minimum(ddrf,self.dcut)/float(self.pd[3:]))**self.expp)
        else:
            ddlf=(1-np.minimum(ddlf,self.dcut)/self.dcut)**float(self.pd)
            ddrf=(1-np.minimum(ddrf,self.dcut)/self.dcut)**float(self.pd)
            
        
        if self.subsample:
           
            if dplf.shape[0]>self.nop:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=False)
            else:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=True)
                
            if self.half:
                sn=int(self.nop/2)
            else:
                sn=int(self.nop)
            if dprf.shape[0]>self.nop:
                
                rs=np.random.choice(dprf.shape[0], sn, replace=False)
            else:
                rs=np.random.choice(dprf.shape[0], sn, replace=True)
            dprf=dprf[rs,:]
            dlrf=dlrf[rs]
            ddrf=ddrf[rs]
            dplf=dplf[ls,:]
            dllf=dllf[ls]
            ddlf=ddlf[ls]
        
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,5))
#             dprf=dprf.reshape((dprf.shape[0],-1,5))
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            if self.normsize:
                dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/self.maxradius
                dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/self.maxradius
            
            dplf[:,3]=np.maximum(np.minimum(dplf[:,3],50),-50)/50
            dprf[:,3]=np.maximum(np.minimum(dprf[:,3],50),-50)/50
            
            dplf[:,4]=dplf[:,4]/4.5
            dprf[:,4]=dprf[:,4]/4.5
            
            dplf[:,[5,6,7]] = dplf[:,[5,6,7]] / np.expand_dims(np.sum(dplf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            dprf[:,[5,6,7]] = dprf[:,[5,6,7]] / np.expand_dims(np.sum(dprf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            
            
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,eroi] = dplf[:,eroi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,eroi] = dprf[:,eroi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ddlf).float(),torch.from_numpy(ddrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)

class ProteinDatasetSamMasifPP(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,normsize=0,subsample=True,mul=30,exppower=1,folder='masif',nop=2048,dcut=10,powerd=1,hf=0,kl=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
#         self.filelist = json.load(open(file, 'r'))
        with open(file, "rb") as input_file:
            self.filelist = list(pickle.load(input_file).keys())
        self.mul=mul
        self.folder=folder
        self.nop=nop
        self.pd=powerd
        self.dcut=dcut
        self.half=hf
        self.expp=exppower
        self.normsize=normsize
        self.maxradius=100
        self.pkl=kl
        


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
#         plf=self.folder+'/pts2/'+self.filelist[cindex][-6:-1]+'l.pts'
#         prf=self.folder+'/pts2/'+self.filelist[cindex][-6:-1]+'r.pts'
        
#         llf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.seg'
#         lrf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.seg'
        
#         if self.pkl:
#             dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.ssseg'
#             drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.ssseg'
#         else:
#             dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.sseg'
#             drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.sseg'

        plf=self.folder+'/pts3/'+self.filelist[cindex]+'-l.pts'
        prf=self.folder+'/pts3/'+self.filelist[cindex]+'-r.pts'
        
        llf=self.folder+'/pts/'+self.filelist[cindex]+'-l.seg'
        lrf=self.folder+'/pts/'+self.filelist[cindex]+'-r.seg'
        
        dlf=self.folder+'/pts/'+self.filelist[cindex]+'-l.ssseg'
        drf=self.folder+'/pts/'+self.filelist[cindex]+'-r.ssseg'
        
        tdplf=np.loadtxt(plf)
        tdprf=np.loadtxt(prf)
        
        if tdplf.shape[0]>tdprf.shape[0]:
            
            dplf=tdplf
            dprf=tdprf
        
            dllf=np.loadtxt(llf)
            dlrf=np.loadtxt(lrf)
            
            ddlf=np.loadtxt(dlf)
            ddrf=np.loadtxt(drf)
        else:
            dplf=tdprf
            dprf=tdplf
        
            dllf=np.loadtxt(lrf)
            dlrf=np.loadtxt(llf)
            
            ddlf=np.loadtxt(drf)
            ddrf=np.loadtxt(dlf)
            
        if self.pd==None:
            pass
        elif self.pd[0:3]=='exp':
            ddlf=np.exp(-(np.minimum(ddlf,self.dcut)/float(self.pd[3:]))**self.expp)
            ddrf=np.exp(-(np.minimum(ddrf,self.dcut)/float(self.pd[3:]))**self.expp)
        else:
            ddlf=(1-np.minimum(ddlf,self.dcut)/self.dcut)**float(self.pd)
            ddrf=(1-np.minimum(ddrf,self.dcut)/self.dcut)**float(self.pd)
            
        
        if self.subsample:
           
            if dplf.shape[0]>self.nop:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=False)
            else:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=True)
                
            if self.half:
                sn=int(self.nop/2)
            else:
                sn=int(self.nop)
            if dprf.shape[0]>self.nop:
                
                rs=np.random.choice(dprf.shape[0], sn, replace=False)
            else:
                rs=np.random.choice(dprf.shape[0], sn, replace=True)
            dprf=dprf[rs,:]
            dlrf=dlrf[rs]
            ddrf=ddrf[rs]
            dplf=dplf[ls,:]
            dllf=dllf[ls]
            ddlf=ddlf[ls]
        
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,5))
#             dprf=dprf.reshape((dprf.shape[0],-1,5))
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            if self.normsize:
                dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/self.maxradius
                dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/self.maxradius
            
            dplf[:,3]=np.maximum(np.minimum(dplf[:,3],50),-50)/50
            dprf[:,3]=np.maximum(np.minimum(dprf[:,3],50),-50)/50
            
            dplf[:,4]=dplf[:,4]/4.5
            dprf[:,4]=dprf[:,4]/4.5
            
            dplf[:,[5,6,7]] = dplf[:,[5,6,7]] / np.expand_dims(np.sum(dplf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            dprf[:,[5,6,7]] = dprf[:,[5,6,7]] / np.expand_dims(np.sum(dprf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            
            
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,eroi] = dplf[:,eroi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,eroi] = dprf[:,eroi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ddlf).float(),torch.from_numpy(ddrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)

class ProteinDatasetSamMasif(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,subsample=True,mul=30,folder='masif',nop=2048,normsize=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
        with open(file, "rb") as input_file:
            self.filelist = list(pickle.load(input_file).keys())
        self.mul=mul
        self.folder=folder
        self.nop=nop
        self.normsize=normsize


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
        plf=self.folder+'/pts/'+self.filelist[cindex]+'-l.pts'
        prf=self.folder+'/pts/'+self.filelist[cindex]+'-r.pts'
        
        llf=self.folder+'/pts/'+self.filelist[cindex]+'-l.seg'
        lrf=self.folder+'/pts/'+self.filelist[cindex]+'-r.seg'
        
        dlf=self.folder+'/pts/'+self.filelist[cindex]+'-l.ssseg'
        drf=self.folder+'/pts/'+self.filelist[cindex]+'-r.ssseg'
        
        
        dplf=np.loadtxt(plf)
        dprf=np.loadtxt(prf)
        dllf=np.loadtxt(llf)
        dlrf=np.loadtxt(lrf)
        ddlf=np.loadtxt(dlf)
        ddrf=np.loadtxt(drf)
        
        ddlf=np.exp(-np.minimum(ddlf,20)/2)
        ddrf=np.exp(-np.minimum(ddrf,20)/2)
        
        if self.subsample:
           
            if dplf.shape[0]>self.nop:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=False)
            else:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=True)
            if dprf.shape[0]>self.nop:
                
                rs=np.random.choice(dprf.shape[0], self.nop, replace=False)
            else:
                rs=np.random.choice(dprf.shape[0], self.nop, replace=True)
            dprf=dprf[rs,:]
            dlrf=dlrf[rs]
            ddrf=ddrf[rs]
            dplf=dplf[ls,:]
            dllf=dllf[ls]
            ddlf=ddlf[ls]
        
        if self.centroid:
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            
            if self.normsize:
#                 rl=np.max(np.sum(dplf[:,[0,1,2]]**2,0)**0.5)
#                 rr=np.max(np.sum(dprf[:,[0,1,2]]**2,0)**0.5)
#                 rall=max(rl,rr)

                dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/100
                dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/100
            
        
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+6 for i in roi]
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+6 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ddlf).float(),torch.from_numpy(ddrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices




