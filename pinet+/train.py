import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch.optim.lr_scheduler as lr_sched


from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG

from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG



import random
import numpy as np

import sys
import math

from sklearn.metrics import precision_score,recall_score,roc_curve,auc,average_precision_score,balanced_accuracy_score

from model_pu import PointNetDenseClsRes, feature_transform_regularizer

from sklearn.metrics import f1_score,roc_curve,auc,roc_auc_score

from modules import MHA_self_pos,MHA_cross_pos,MHA_cross_pos2,MHA_cross_pos3

import proteindataset

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

num_gpu=[0]
Batchsize=int(32/4)*2
nop=4096*2*2
# nop=4096
# nop=2048

# Batchsize=1
abg=1
gcv=0
presubsam=True
# presubsam=False
vs=0
multi=10
mepoch=100
op='adam'
# tf='dbdapbsfix'
tf='pinetrepofix'
labelminus=0
# indim=14
indim=14
stepsize=20
gamma=0.3
act='l'
lrate=1e-3
# pd=None
# pd='exp0.5'
pd='exp2'
# pd='9'
exppower=1
dcut=20
half=0

lr_clip = 1e-5
bnm_clip = 1e-2

head=8

# sa=0
# ca=4
ln=1
df=0.1
# log_name='pinet2vs'+str(vs)+'_'+str(nop)+'lr'+str(lrate)+'_s'+str(stepsize)+'g'+str(gamma)+'_b'+str(Batchsize)+'_mul'+str(multi)+'_'+op+'_indim_'+str(indim)+'_'+act+'_p'+str(pd)+'dc'+str(dcut)+'h'+str(half)

msf='dbd'
# msf=''

if msf=='masif':
    tf='masif'
    indim=9
    
if msf=='masif++':
    tf='masif'

version='v6'
# v0: 0,1,1,1
# v1:0,1,0,0
# v2:1,0,0,0
# v3:0,0,0,0
# v4:1,1,1,1 cat
# v5: v2+pad depreciated
# v6: v2+2pad
opv=1
ableself=0
normsize=1
tl=0
nopos=0
finetune=1

log_name='pinet2'+version+msf+'_ns'+str(normsize)+'b'+str(Batchsize)+'p'+str(nop)+'h'+str(head)+'mul'+str(multi)+'op'+str(opv)+'_indim'+str(indim)+'_sa'+str(ableself)+str(pd)+'p'+str(exppower)+'hf'+str(half)+'np'+str(nopos)+'ft'+str(finetune)
save_path='/home/bowen/'+log_name
sycn=False

para={}        
para['model.use_xyz']=True
para['optimizer.weight_decay']=0.0
para['optimizer.lr']=lrate
para['optimizer.lr_decay']=0.5
para['optimizer.bn_momentum']=0.5
para['optimizer.bnm_decay']=0.5
# para['optimizer.decay_step']=3e5
para['optimizer.decay_step']=3e5
para['num_points']=nop
para['epochs']=200
para['batch_size']=Batchsize

class pinet2(PointNet2ClassificationSSG):

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        
        c_in = indim-3
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        c_out_3 = 512 + 512
        
#         emb=[c_out_1,c_out_2,c_out_3]
        if version=='v0':
            emb=[512,512,1024]
            ffn=[a*3 for a in emb]
            ca=len(emb)
            sa=len(emb)
            self.ir=[-3,-2,-1]
            
        if version=='v1':
            emb=[512]
            ffn=[a*3 for a in emb]
            ca=len(emb)
            sa=len(emb)
            self.ir=[-3]
            
        if version in ['v2','v5','v6']:
            emb=[256]
            ffn=[a*3 for a in emb]
            ca=len(emb)
            sa=len(emb)
            self.ir=[-4]
        if version=='v3':
            self.ir=[]
            
        if version=='v4':
            emb=[256,512,512,1024]
            ffn=[int(a/2) for a in emb]
            ca=len(emb)
            sa=len(emb)
            self.ir=[-4,-3,-2,-1]
            
        if ableself:
            self.mha_s=[]
            for i in range(sa):
                self.mha_s.append(MHA_self_pos(emb[i],head,ffn[i],ln,df))
            self.mha_s = nn.ModuleList(self.mha_s)
            
        if len(self.ir)>0:
            self.mha_c=[]
            for i in range(ca):
                if ableself:
                    self.mha_c.append(MHA_cross_pos(emb[i],head,ffn[i],ln,df))
                elif version=='v4':
                    self.mha_c.append(MHA_cross_pos3(emb[i],head,ffn[i],ln,df))
#                 elif nopos:
#                     self.mha_c.append(MHA_cross_pos2(emb[i],head,ffn[i],ln,df,nopos=nopo))
                else:
                    self.mha_c.append(MHA_cross_pos2(emb[i],head,ffn[i],ln,df,nopos=nopos))
            self.mha_c = nn.ModuleList(self.mha_c)

        self.FP_modules = nn.ModuleList()
        if version=='v4':
            self.FP_modules.append(PointnetFPModule(mlp=[256 + indim-3+32, 128, 128]))
            self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0+32, 256, 256]))
            self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1+32, 512, 512]))
            self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2+32, 512, 512]))
        else:
            self.FP_modules.append(PointnetFPModule(mlp=[256 + indim-3, 128, 128]))
            self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
            self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
            self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, kernel_size=1),
        )
        
        if finetune==1:
            cp=torch.load('/home/bowen/pinet2v6masif++_ns1b32p8192h8mul5op1_indim14_sa0exp2p1hf0np0/pn-val-epoch=09-val_total=0.983.ckpt')            
            self.load_state_dict(cp['state_dict'])
        

    def forward(self, pointcloud0,pointcloud1):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz0, features0 = self._break_up_pc(pointcloud0)
        xyz1, features1 = self._break_up_pc(pointcloud1)

        l_xyz0, l_features0 = [xyz0], [features0]
        l_xyz1, l_features1 = [xyz1], [features1]
        
        for i in range(len(self.SA_modules)):
            li_xyz0, li_features0 = self.SA_modules[i](l_xyz0[i], l_features0[i])
            li_xyz1, li_features1 = self.SA_modules[i](l_xyz1[i], l_features1[i])
            
            l_xyz0.append(li_xyz0)
            l_xyz1.append(li_xyz1)
            
            l_features0.append(li_features0)
            l_features1.append(li_features1)
            
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            if i in self.ir:
                if ableself:
                    emb0,_=self.mha_s[self.ir.index(i)](l_xyz0[i], l_features0[i])
                    emb1,_=self.mha_s[self.ir.index(i)](l_xyz1[i], l_features1[i])
                
                    sour0,_=self.mha_c[self.ir.index(i)](l_xyz0[i], emb0,l_xyz1[i], emb1)
                    sour1,_=self.mha_c[self.ir.index(i)](l_xyz1[i], emb1,l_xyz0[i], emb0)
                else:
#                     print(l_features0[i].size())
#                     print(l_features1[i].size())
                    if version=='v5':
                        p1d = (0, 1)
                        sour0,_=self.mha_c[self.ir.index(i)](l_xyz0[i], l_features0[i],
                                                             l_xyz1[i], F.pad(l_features1[i], p1d, "constant", 0))
                        sour1,_=self.mha_c[self.ir.index(i)](l_xyz1[i], l_features1[i],
                                                             l_xyz0[i], F.pad(l_features0[i], p1d, "constant", 0))
                    elif version=='v6':
                        p1d = (0, 1)
                        p1dp = (0,0,0, 1)
                        sour0,_=self.mha_c[self.ir.index(i)](l_xyz0[i], l_features0[i],
                                                             F.pad(l_xyz1[i], p1dp, "constant", 0),
                                                             F.pad(l_features1[i], p1d, "constant", 0))
                        sour1,_=self.mha_c[self.ir.index(i)](l_xyz1[i], l_features1[i],
                                                             F.pad(l_xyz0[i], p1dp, "constant", 0), 
                                                             F.pad(l_features0[i], p1d, "constant", 0))
                    else:
                        sour0,_=self.mha_c[self.ir.index(i)](l_xyz0[i], l_features0[i],l_xyz1[i], l_features1[i])
                        sour1,_=self.mha_c[self.ir.index(i)](l_xyz1[i], l_features1[i],l_xyz0[i], l_features0[i])
        
#                 sour0,_=self.mha_c[i](l_xyz0[i], l_features0[i],l_xyz1[i], l_features1[i])
#                 sour1,_=self.mha_c[i](l_xyz1[i], l_features1[i],l_xyz0[i], l_features0[i])
            

            # B C L->L B C
#                 sour0=sour0.contiguous()
#                 sour1=sour1.contiguous()
                if version!='v4':
                    l_features0[i - 1] = self.FP_modules[i](
                        l_xyz0[i - 1], l_xyz0[i], l_features0[i - 1], sour0.permute(1,2,0).contiguous()
                    )
                    l_features1[i - 1] = self.FP_modules[i](
                        l_xyz1[i - 1], l_xyz1[i], l_features1[i - 1], sour1.permute(1,2,0).contiguous()
                    )
                else:

                    l_features0[i - 1] = self.FP_modules[i](
                        l_xyz0[i - 1], l_xyz0[i], l_features0[i - 1], torch.cat((l_features0[i],sour0.permute(1,2,0).contiguous()),1)
                    )
                    l_features1[i - 1] = self.FP_modules[i](
                        l_xyz1[i - 1], l_xyz1[i], l_features1[i - 1], torch.cat((l_features1[i],sour1.permute(1,2,0).contiguous()),1)
                    )
                    
                    
            else:
                l_features0[i - 1] = self.FP_modules[i](
                    l_xyz0[i - 1], l_xyz0[i], l_features0[i - 1], l_features0[i]
                )


                l_features1[i - 1] = self.FP_modules[i](
                    l_xyz1[i - 1], l_xyz1[i], l_features1[i - 1], l_features1[i]
                )



        return self.fc_lyaer(torch.cat((l_features0[0],l_features1[0]),2))
    

    def training_step(self, batch, batch_idx):

        
        dl,dr,ll,lr,sdl,sdr = batch
        
#         dr = dr.transpose(2, 1)
#         dl = dl.transpose(2, 1)
        
#         dr=dr[:,0:indim,:]
#         dl=dl[:,0:indim,:]
        dr=dr[:,:,0:indim]
        dl=dl[:,:,0:indim]


        py=self(dr,dl).view(-1,1)
        if labelminus:
            y=torch.cat((lr,ll),dim=1).view(-1,1)-1
        else:
            y=torch.cat((lr,ll),dim=1).view(-1,1)
            sy=torch.cat((sdr,sdl),dim=1).view(-1,1)
        if pd==None:
            loss = 1.0*F.binary_cross_entropy_with_logits(py, y.float(),pos_weight=torch.FloatTensor([(y.size()[0]-float(y.sum()))*1.0/float(y.sum())]).type_as(py))
        else:
            loss = 1.0*F.binary_cross_entropy_with_logits(py, sy.float(),pos_weight=torch.FloatTensor([(y.size()[0]-float(y.sum()))*1.0/float(y.sum())]).type_as(py))
    
        accuracy =roc_auc_score(y.view(1,-1).squeeze().cpu().numpy(),torch.sigmoid(py).view(1,-1).squeeze().detach().cpu().numpy())
        self.log('auc', accuracy, sync_dist=sycn)
            
        self.log('total', loss, sync_dist=sycn)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        dl,dr,ll,lr,sdl,sdr = batch

#         dr = dr.transpose(2, 1)
#         dl = dl.transpose(2, 1)
        
#         dr=dr[:,0:indim,:]
#         dl=dl[:,0:indim,:]
        
        dr=dr[:,:,0:indim]
        dl=dl[:,:,0:indim]
        
        
        py=self(dr,dl).view(-1,1)
        if labelminus:
            y=torch.cat((lr,ll),dim=1).view(-1,1)-1
        else:
            y=torch.cat((lr,ll),dim=1).view(-1,1)        
        loss=F.binary_cross_entropy_with_logits(py.squeeze(),y.type_as(py).squeeze(),pos_weight=(torch.numel(y)-y.sum())/y.sum())
        accuracy =roc_auc_score(y.squeeze().cpu().numpy(),torch.sigmoid(py).squeeze().detach().cpu().numpy())
        self.log('val_auc', accuracy, sync_dist=sycn, on_epoch=True)   
        accuracy =average_precision_score(y.squeeze().cpu().numpy(),torch.sigmoid(py).squeeze().detach().cpu().numpy())
        self.log('val_pr', accuracy, sync_dist=sycn, on_epoch=True)
        bas = balanced_accuracy_score(y.squeeze().cpu().numpy(),torch.sigmoid(py).squeeze().detach().cpu().numpy()>0.5)
        self.log('val_bas', bas, sync_dist=sycn, on_epoch=True)  
        p =precision_score(y.squeeze().cpu().numpy(),torch.sigmoid(py).squeeze().detach().cpu().numpy()>0.5)
        self.log('val_p', p, sync_dist=sycn, on_epoch=True)    
        r =recall_score(y.squeeze().cpu().numpy(),torch.sigmoid(py).squeeze().detach().cpu().numpy()>0.5)
        self.log('val_r', r, sync_dist=sycn, on_epoch=True)    
            
        self.log('val_total', loss)
        

    def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.parameters(),
#             betas=(0.9, 0.98),
#             lr=5e-4,
#             eps=1e-06, 
#             weight_decay=0.01
#         )
        if opv==0:
            if op=='adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=lrate, betas=(0.9, 0.98))
            else:
                optimizer = torch.optim.SGD(self.parameters(), lr=lrate, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

            return [optimizer],[scheduler]
        else:
            lr_lbmd = lambda _: max(
                self.hparams["optimizer.lr_decay"]
                ** (
                    int(
                        self.global_step
                        * self.hparams["batch_size"]
                        / self.hparams["optimizer.decay_step"]
                    )
                ),
                lr_clip / self.hparams["optimizer.lr"],
            )
    #         bn_lbmd = lambda _: max(
    #             self.hparams["optimizer.bn_momentum"]
    #             * self.hparams["optimizer.bnm_decay"]
    #             ** (
    #                 int(
    #                     self.global_step
    #                     * self.hparams["batch_size"]
    #                     / self.hparams["optimizer.decay_step"]
    #                 )
    #             ),
    #             bnm_clip,
    #         )

            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
    #         bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

    #         return [optimizer], [lr_scheduler, bnm_scheduler]
            return [optimizer], [lr_scheduler]




    def prepare_data(self):
        pass

    def train_dataloader(self):
        if msf=='masif':
            train=proteindataset.ProteinDatasetSamMasif('masif/train.pkl',subsample=presubsam,
                                                        normsize=normsize,mul=multi,folder=tf,nop=nop)
        elif msf=='masif++':
            train=proteindataset.ProteinDatasetSamMasifPP('masif/train.pkl',subsample=presubsam,
                                                        normsize=normsize,mul=multi,folder=tf,
                                                          nop=nop,dcut=dcut,powerd=pd,hf=half)
        else:
            train=proteindataset.ProteinDatasetSam('pinet/PInet/dbdapbscon/train_test_split/shuffled_train_file_list_l.json',
                                                 subsample=presubsam,normsize=normsize,mul=multi,
                                                   folder=tf,nop=nop,dcut=dcut,powerd=pd,hf=half)

        return DataLoader(train,batch_size=Batchsize,num_workers=4,shuffle=True,drop_last=True,worker_init_fn=worker_init_fn)
        
    def val_dataloader(self):
        if msf=='masif':
            test=proteindataset.ProteinDatasetSamMasif('masif/test.pkl',subsample=False,
                                                       normsize=normsize,mul=1,folder=tf,nop=nop)
        elif msf=='masif++':
            test=proteindataset.ProteinDatasetSamMasifPP('masif/test.pkl',subsample=False,
                                                       normsize=normsize,mul=1,
                                                         folder=tf,nop=nop,dcut=dcut,powerd=pd,hf=half)
        else:
            test=proteindataset.ProteinDatasetSam('pinet/PInet/dbdapbscon/train_test_split/shuffled_test_file_list_l.json',
                                              subsample=vs,normsize=normsize,mul=1,
                                                  folder=tf,nop=nop,dcut=dcut,powerd=pd,hf=half)
        
        return DataLoader(test,batch_size=1,num_workers=4)
    


tomoon = pinet2(para)

checkpoint_val = ModelCheckpoint(
    monitor='val_total',
    dirpath=save_path,
    filename='pn-val-{epoch:02d}-{val_total:.3f}',
#     period=1,
    save_top_k=1,
    mode='min',
)

checkpoint_train = ModelCheckpoint(
    monitor='total',
    dirpath=save_path,
    filename='pn-train-{epoch:02d}-{total:.3f}',
#     period=1,
    save_top_k=1,
    mode='min',
)

logger = CSVLogger("logs", name=log_name)

trainer = pl.Trainer(
    max_epochs=mepoch,
    gpus=num_gpu, 
#     accelerator='ddp',
#     plugins='ddp_sharded',
#     precision=16,
    accumulate_grad_batches=abg,
    gradient_clip_val=gcv,
    check_val_every_n_epoch=10,
    callbacks=[checkpoint_train,checkpoint_val],
    logger=logger
#     default_root_dir='/home/bowen/glycan/glycp'
)




trainer.fit(tomoon)