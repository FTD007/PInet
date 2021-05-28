import pymol
from pymol import cmd,stored
import os
import sys
import re
import numpy as np
from dx2feature import *
from getResiLabel import *
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors


# pdb to wrl

pdbfile_l=sys.argv[1]
pdbfile_r=sys.argv[2]

train_flag=0
needapbs=0

if len(sys.argv)==4 and sys.argv[3]=='train':
    train_flag=1

cmd.load(pdbfile_l)
cmd.set('surface_quality', '0')
cmd.show_as('surface', 'all')
cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
cmd.save(pdbfile_l[0:4]+'-l.wrl')
cmd.delete('all')

cmd.load(pdbfile_r)
cmd.set('surface_quality', '0')
cmd.show_as('surface', 'all')
cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
cmd.save(pdbfile_r[0:4]+'-r.wrl')
cmd.delete('all')

# wrl to pts

holder = []
normholder =[]
cf=0
nf=0
with open(pdbfile_l[0:4]+'-l.wrl', "r") as vrml:
    for lines in vrml:
        if 'point [' in lines:
            cf=1
        if cf==1:
            a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
            if len(a) == 3:
                holder.append(tuple(map(float, a)))
        if 'vector [' in lines:
            nf=1
        if nf==1:
            a = re.findall("[-0-9]{1}.[0-9]{4}", lines)
            if len(a) == 3:
                normholder.append(tuple(map(float, a)))
                
lcoord=np.array(holder)
# print lcoord

holder = []
normholder =[]
cf=0
nf=0
with open(pdbfile_r[0:4]+'-r.wrl', "r") as vrml:
    for lines in vrml:
        if 'point [' in lines:
            cf=1
        if cf==1:
            a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
            if len(a) == 3:
                holder.append(tuple(map(float, a)))
        if 'vector [' in lines:
            nf=1
        if nf==1:
            a = re.findall("[-0-9]{1}.[0-9]{4}", lines)
            if len(a) == 3:
                normholder.append(tuple(map(float, a)))
                
rcoord=np.array(holder)

lcoord=np.unique(lcoord,axis=0)
rcoord=np.unique(rcoord,axis=0)

print(lcoord)


if train_flag:

    tol=np.array([2,2,2])

    contact = (np.abs(np.asarray(lcoord[:, None]) - np.asarray(rcoord))<tol).all(2).astype(np.int)

    llabel=np.max(contact,axis=1)
    rlabel=np.max(contact,axis=0)
    
    np.savetxt(pdbfile_l[0:4]+'-l.seg',llabel)
    np.savetxt(pdbfile_r[0:4]+'-r.seg',rlabel)
    
# pdb 2 pqr
# pdb2pqr='/path/to/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
# apbsflag='--whitespace --ff=amber -v --apbs-input'
# apbs='/path/to/apbs-pdb2pqr/bin/apbs'

if needapbs:
    pdb2pqr='/dartfs-hpc/rc/home/w/f00355w/Bdai/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
    apbsflag='--whitespace --ff=amber -v --apbs-input'
    apbs='/dartfs-hpc/rc/home/w/f00355w/Bdai/apbs-pdb2pqr/bin/apbs'


    try:
        os.system(pdb2pqr+' '+apbsflag+' '+pdbfile_l[0:4]+'-l.pdb'+' '+pdbfile_l[0:4]+'-l.pqr')
    except:
        print('error when pdb2pqr l: '+pdbfile_l[0:4])

    try:
        os.system(pdb2pqr+' '+apbsflag+' '+pdbfile_r[0:4]+'-r.pdb'+' '+pdbfile_r[0:4]+'-r.pqr')
    except:
        print('error when pdb2pqr r: '+pdbfile_r[0:4])

    try:
        os.system(apbs+' '+pdbfile_l[0:4]+'-l.in')
    except:
        print('error when abps l: '+pdbfile_l[0:4])

    try:
        os.system(apbs+' '+pdbfile_r[0:4]+'-r.in')
    except:
        print('error when abps r: '+pdbfile_r[0:4])
    
    
# add apbs feature

# newdick, labeldick, ab, ag = getcontactbyabag('/path/to/complex/pdb/files', ab=pdbdic[key][0], ag=pdbdic[key][1])
centroid_l, labelsl = gethydro(pdbfile_l)
centroid_r, labelsr = gethydro(pdbfile_r)


centroid_l = np.array(centroid_l)
centroid_r = np.array(centroid_r)

hlabell = np.transpose(np.asarray(labelsl[0]))
hlabelr = np.transpose(np.asarray(labelsr[0]))

clfl = neighbors.KNeighborsClassifier(3)
clfr = neighbors.KNeighborsClassifier(3)

clfl.fit(centroid_l,hlabell*10)
clfr.fit(centroid_r,hlabelr*10)

distl,indl=clfl.kneighbors(lcoord)
distr,indr= clfr.kneighbors(rcoord)

apbsl=open(pdbfile_l[0:4]+'-l.pqr.dx','r')
apbsr = open(pdbfile_r[0:4] + '-r.pqr.dx','r')

gl, orl, dl, vl = parsefile(apbsl)
gr, orr, dr, vr = parsefile(apbsr)


avl=findvalue(lcoord, gl, orl, dl, vl)
avr = findvalue(rcoord, gr, orr, dr, vr)



lpred=np.sum(hlabell[indl]*distl,1)/np.sum(distl,1)
rpred = np.sum(hlabelr[indr] * distr, 1) / np.sum(distr, 1)


np.savetxt(pdbfile_l[0:4]+'-l.pts',np.concatenate((lcoord,np.expand_dims(avl,1),np.expand_dims(lpred,1)),axis=1))
np.savetxt(pdbfile_r[0:4]+'-r.pts',np.concatenate((rcoord, np.expand_dims(avr,1),np.expand_dims(rpred,1)),axis=1))
