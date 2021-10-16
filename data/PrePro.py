import pymol
from pymol import cmd,stored
import os
import subprocess
import sys
import re
import numpy as np
from dx2feature import *
from getResiLabel import *
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors


# pdb to wrl

# pdbfile_l=sys.argv[1]
# pdbfile_r=sys.argv[2]

def prepro(pdb,pdbpair,not_skip_pymol=0,train_flag=0,needapbs=0,pf='01-benchmark_pdbs/',wf='wrl-fixed/',af='./',ptsf='pts2/',segf='seg/'):
    if pdbpair!=None:
        pdbfile_l=pdbpair[0][0:-4]+'.pdb'
        pdbfile_r=pdbpair[1][0:-4]+'.pdb'
    else:
        pdbfile_l=pdb+'_l.pdb'
        pdbfile_r=pdb+'_r.pdb'
        
    if not_skip_pymol:

        cmd.load(pf+pdbfile_l)
        cmd.set('surface_quality', '0')
        cmd.show_as('surface', 'all')
        cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
        cmd.save(wf+pdbfile_l[0:4]+'-l.wrl')
        cmd.delete('all')

        cmd.load(pf+pdbfile_r)
        cmd.set('surface_quality', '0')
        cmd.show_as('surface', 'all')
        cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
        cmd.save(wf+pdbfile_r[0:4]+'-r.wrl')
        cmd.delete('all')

        # wrl to pts

        holder = []
        normholder =[]
        cf=0
        nf=0
        with open(wf+pdbfile_l[0:4]+'-l.wrl', "r") as vrml:
            for lines in vrml:
                if 'point [' in lines:
                    cf=1
                if cf==1:
                    if ']' not in lines:
                        a = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                        if len(a) == 3:
                            holder.append(tuple(map(float, a)))
                    else:
                        cf=0
                if 'vector [' in lines:
                    nf=1
                    cf=0
                if nf==1:
                    if ']' not in lines:
                        a = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                        if len(a) == 3:
                            normholder.append(tuple(map(float, a)))
                    else:
                        nf=0

        lcoord=np.array(holder)
        lsnorm=np.array(normholder)
        # print lcoord

        holder = []
        normholder =[]
        cf=0
        nf=0
        with open(wf+pdbfile_r[0:4]+'-r.wrl', "r") as vrml:
            for lines in vrml:
                if 'point [' in lines:
                    cf=1
                if cf==1:
                    if ']' not in lines:
                        a = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                        if len(a) == 3:
                            holder.append(tuple(map(float, a)))
                    else:
                        cf=0
                if 'vector [' in lines:
                    nf=1
                    cf=0
                if nf==1:
                    if ']' not in lines:
                        a = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                        if len(a) == 3:
                            normholder.append(tuple(map(float, a)))
                    else:
                        nf=0

        rcoord=np.array(holder)
        rsnorm=np.array(normholder)

        lcoord,li=np.unique(lcoord,axis=0,return_index=True)
        rcoord,ri=np.unique(rcoord,axis=0,return_index=True)

        lsnorm=lsnorm[li,:]
        rsnorm=rsnorm[ri,:]
    else:
#         datal=np.loadtxt('pts2/'+pdb+'-l.pts')
#         datar=np.loadtxt('pts2/'+pdb+'-r.pts')
        datal=np.loadtxt('pts/'+pdb+'-l.pts')
        datar=np.loadtxt('pts/'+pdb+'-r.pts')

        labell=np.loadtxt('pts/'+pdb+'-l.seg')
        labelr=np.loadtxt('pts/'+pdb+'-r.seg')

        lcoord=datal[:,0:3]
        rcoord=datar[:,0:3]
        
        lsnorm=datal[:,6:]
        rsnorm=datar[:,6:]
#     print(lcoord)


    if train_flag:

        tol=np.array([2,2,2])

        contact = (np.abs(np.asarray(lcoord[:, None]) - np.asarray(rcoord))<tol).all(2).astype(np.int)

        llabel=np.max(contact,axis=1)
        rlabel=np.max(contact,axis=0)

        np.savetxt(segf+pdbfile_l[0:4]+'-l.seg',llabel)
        np.savetxt(segf+pdbfile_r[0:4]+'-r.seg',rlabel)

    # pdb 2 pqr
    # pdb2pqr='/path/to/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
    # apbsflag='--whitespace --ff=amber -v --apbs-input'
    # apbs='/path/to/apbs-pdb2pqr/bin/apbs'

    if needapbs:
    #     pdb2pqr='/dartfs-hpc/rc/home/w/f00355w/Bdai/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
        pdb2pqr='/home/bowen/pinetrepofix/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
        apbsflag='--whitespace --ff=amber -v --apbs-input'
        apbs='/home/bowen/pinetrepofix/apbs-pdb2pqr/bin/apbs'


        try:
#             os.system(pdb2pqr+' '+apbsflag+' '+pdbfile_l[0:4]+'_l.pdb'+' '+pdbfile_l[0:4]+'-l.pqr')
            os.system(pdb2pqr+' '+apbsflag+' '+pf+pdbfile_l+' '+pdbfile_l[0:4]+'-l.pqr')
        except:
            print('error when pdb2pqr l: '+pdbfile_l[0:4])

        try:
#             os.system(pdb2pqr+' '+apbsflag+' '+pdbfile_r[0:4]+'_r.pdb'+' '+pdbfile_r[0:4]+'-r.pqr')
            os.system(pdb2pqr+' '+apbsflag+' '+pf+pdbfile_r+' '+pdbfile_r[0:4]+'-r.pqr')
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
    centroid_l, labelsl = getlabels(pf+pdbfile_l)
    centroid_r, labelsr = getlabels(pf+pdbfile_r)

#     print(labelsl)
#     print(labelsr)

    rlabel = np.transpose(np.asarray(labelsr))
    llabel = np.transpose(np.asarray(labelsl))



    centroid_l = np.array(centroid_l)
    centroid_r = np.array(centroid_r)

    # hlabell = np.transpose(np.asarray(labelsl[0]))
    # hlabelr = np.transpose(np.asarray(labelsr[0]))
    nn=1
    clfl = neighbors.KNeighborsClassifier(nn)
    clfr = neighbors.KNeighborsClassifier(nn)

    # clfl.fit(centroid_l,hlabell*10)
    # clfr.fit(centroid_r,hlabelr*10)
    clfl.fit(centroid_l,llabel*10)
    clfr.fit(centroid_r,rlabel*10)


    distl,indl=clfl.kneighbors(lcoord)
    distr,indr= clfr.kneighbors(rcoord)
#     print(distl.shape)
#     print(indl.shape)

    with open(af+pdbfile_l[0:4]+'-l.pqr.dx','r') as apbsl:
        gl, orl, dl, vl = parsefile(apbsl)
    with open(af+pdbfile_r[0:4] + '-r.pqr.dx','r') as apbsr:
        gr, orr, dr, vr = parsefile(apbsr)
#     gl, orl, dl, vl = parsefile(apbsl)
#     gr, orr, dr, vr = parsefile(apbsr)


#     avl = findvalueall(lcoord, gl, orl, dl, vl)
#     avr = findvalueall(rcoord, gr, orr, dr, vr)
    
    avl = findvalueallm(lcoord, gl, orl, dl, vl)
    avr = findvalueallm(rcoord, gr, orr, dr, vr)

    distl=np.exp(-distl**2)
    distr=np.exp(-distr**2)

    lpred = np.sum(llabel[indl] * np.expand_dims(distl, 2), 1) / np.expand_dims(np.sum(distl, 1), 1)
    rpred = np.sum(rlabel[indr] * np.expand_dims(distr, 2), 1) / np.expand_dims(np.sum(distr, 1), 1)
    
#     print(lcoord.shape)
#     print(avl.shape)
#     print(lpred.shape)

    
#     np.savetxt(ptsf+pdbfile_l[0:4]+'-l.pts',np.concatenate((lcoord,lsnorm,np.transpose(avl),lpred),axis=1), fmt='%1.6e')
#     np.savetxt(ptsf+pdbfile_r[0:4]+'-r.pts',np.concatenate((rcoord,rsnorm,np.transpose(avr),rpred),axis=1), fmt='%1.6e')
    sdatal=np.concatenate((lcoord,lsnorm,np.transpose(avl),lpred),axis=1)
    sdatar=np.concatenate((rcoord,rsnorm,np.transpose(avr),rpred),axis=1)
    np.savetxt(ptsf+pdbfile_l[0:4]+'-l.pts',
               np.hstack((sdatal[:,0:3],np.expand_dims(sdatal[:,6],1),
                          np.expand_dims(sdatal[:,10],1),sdatal[:,3:6],sdatal[:,7:10],sdatal[:,11:])), fmt='%1.6e')
    np.savetxt(ptsf+pdbfile_r[0:4]+'-r.pts',
                   np.hstack((sdatar[:,0:3],np.expand_dims(sdatar[:,6],1),
                              np.expand_dims(sdatar[:,10],1),sdatar[:,3:6],sdatar[:,7:10],sdatar[:,11:])), fmt='%1.6e')
    
    subprocess.check_output('rm '+pdbfile_r[0:4]+'*', shell=True)