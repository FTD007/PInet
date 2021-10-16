import os
import csv
import numpy as np
from dx2feature import *
file=open('table_BM5c.csv','r')
##put your csv table listing pdb pairs

cr=csv.reader(file)
pdbdic={}
for row in cr:
    
    crow = row[0]
    pdb=crow[0:4]
    i1=crow.find('_')
    i2=crow.find(':')
    i3=crow.find(' ')
    if i3>0:
        pdbdic[pdb]=[crow[i1+1:i2],crow[i2+1:i3]]
    else:
        pdbdic[pdb] = [crow[i1 + 1:i2], crow[i2 + 1:]]

print len(pdbdic)
from getResiLabel import *
from sklearn import neighbors

orifolder='/path/to/data/from/matlab/'
newfolder='/save/to/this/folder/'
if not os.path.exists(newfolder+'lf/points'):
    os.mkdir(newfolder + 'lf')
    os.mkdir(newfolder+'lf/points')
    os.mkdir(newfolder + 'lf/points_label')
    os.mkdir(newfolder + 'rf')
    os.mkdir(newfolder + 'rf/points')
    os.mkdir(newfolder + 'rf/points_label')
apbsfolder='/apbs/output/folder/'
for key in pdbdic:
    # print key
    # d1,d2,ab,ag=getcontactbyabag('/Users/bowendai/Documents/Topoprotein/bench/selectpdb/','clean'+key+'-l-r.pdb',ab=pdbdic[key][0],ag=pdbdic[key][1])
    newdick, labeldick, ab, ag = getcontactbyabag('/path/to/complex/pdb/files', ab=pdbdic[key][0], ag=pdbdic[key][1])
    # print d1
    # print d2
    # print newdick
    # print labeldick
    # break
    # lfile=np.loadtxt(orifolder+'lf/points/'+key+'_l.pts')
    # rfile = np.loadtxt(orifolder + 'rf/points/' + key + '_r.pts')
    lfile = np.loadtxt(orifolder + 'all/points/' + key + '-l.pts')[:,0:3]
    rfile = np.loadtxt(orifolder + 'all/points/' + key + '-r.pts')[:,0:3]
    # print lfile.shape

    # newdick, labeldick, ab, ag = getcontactbyabag(pdbfolder, pdb + '.pdb', d=4.5, ab=ab, ag=ag)
    rcoord = np.asarray(newdick['r'])
    rlabel = np.transpose(np.asarray(labeldick['r']))
    # print rcoord.shape

    lcoord = np.asarray(newdick['l'])
    llabel = np.transpose(np.asarray(labeldick['l']))

    clfl = neighbors.KNeighborsClassifier(3)
    clfr = neighbors.KNeighborsClassifier(3)

    clfl.fit(lcoord,llabel*10)
    clfr.fit(rcoord,rlabel*10)
    # print llabel.shape

    distl,indl=clfl.kneighbors(lfile)
    distr,indr= clfr.kneighbors(rfile)
    # print distl.shape
    # print np.sum(llabel[indl]*np.expand_dims(distl,2),2).shape

    apbsl=open(apbsfolder+key+'-l.pqr.dx','r')
    apbsr = open(apbsfolder + key + '-r.pqr.dx','r')

    gl, orl, dl, vl = parsefile(apbsl)
    gr, orr, dr, vr = parsefile(apbsr)

    # print gl, orl, dl
    # print orl+gl*dl
    # print np.amax(lfile,axis=0)
    # print np.amin(lfile,axis=0)


    avl=findvalue(lfile, gl, orl, dl, vl)
    avr = findvalue(rfile, gr, orr, dr, vr)

    # lpred=clfl.predict(lfile)/10
    # rpred=clfr.predict(rfile)/10
    lpred=np.sum(llabel[indl]*np.expand_dims(distl,2),1)/np.expand_dims(np.sum(distl,1),1)/10
    rpred = np.sum(rlabel[indr] * np.expand_dims(distr, 2), 1) / np.expand_dims(np.sum(distr, 1), 1)/10
    # print lpred.shape

    # print lfile.shape
    # print np.expand_dims(avl,2).shape
    # print lpred[:,1].shape

    np.savetxt(newfolder+'lf/points/'+key+'-l.pts',np.concatenate((lfile,np.expand_dims(avl,2),np.expand_dims(lpred[:,0],2)),axis=1))
    np.savetxt(newfolder + 'rf/points/' + key + '-r.pts', np.concatenate((rfile, np.expand_dims(avr,2),np.expand_dims(rpred[:,0],2)),axis=1))
    # print 1
