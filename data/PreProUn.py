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

def prepro(pdb,not_skip_pymol=1,train_flag=0,needapbs=1,pf='',wf='',af='./',ptsf='',rm=1):
    pdbfile_l=pdb
        
    if not_skip_pymol:

        cmd.load(pf+pdbfile_l)
        cmd.set('surface_quality', '0')
        cmd.show_as('surface', 'all')
        cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
        cmd.save(wf+pdbfile_l[0:4]+'-l.wrl')
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

        lcoord,li=np.unique(lcoord,axis=0,return_index=True)

        lsnorm=lsnorm[li,:]
    else:

        datal=np.loadtxt('pts/'+pdb+'-l.pts')

        labell=np.loadtxt('pts/'+pdb+'-l.seg')

        lcoord=datal[:,0:3]
        
        lsnorm=datal[:,6:]

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
            os.system(apbs+' '+pdbfile_l[0:4]+'-l.in')
        except:
            print('error when abps l: '+pdbfile_l[0:4])


    # add apbs feature

    centroid_l, labelsl = getlabels(pf+pdbfile_l)

    llabel = np.transpose(np.asarray(labelsl))

    centroid_l = np.array(centroid_l)

    nn=1
    clfl = neighbors.KNeighborsClassifier(nn)

    clfl.fit(centroid_l,llabel*10)


    distl,indl=clfl.kneighbors(lcoord)


    with open(af+pdbfile_l[0:4]+'-l.pqr.dx','r') as apbsl:
        gl, orl, dl, vl = parsefile(apbsl)
    
    avl = findvalueallm(lcoord, gl, orl, dl, vl)

    distl=np.exp(-distl**2)

    lpred = np.sum(llabel[indl] * np.expand_dims(distl, 2), 1) / np.expand_dims(np.sum(distl, 1), 1)

    sdatal=np.concatenate((lcoord,lsnorm,np.transpose(avl),lpred),axis=1)
    np.savetxt(ptsf+pdbfile_l[0:4]+'-l.pts',
               np.hstack((sdatal[:,0:3],np.expand_dims(sdatal[:,6],1),
                          np.expand_dims(sdatal[:,10],1),sdatal[:,3:6],sdatal[:,7:10],sdatal[:,11:])), fmt='%1.6e')
    if rm:
        subprocess.check_output('rm '+pdbfile_l[0:4]+'*', shell=True)