# # from gridData import Grid
# # g=Grid('/Users/bowendai/Documents/nn/pointprotein/4jr9.pqr.dx')
# from io import StringIO
# # from ast import literal_eval
# import os
import numpy as np
# file=open('/Users/bowendai/Documents/nn/pointprotein/4jr9.pqr.dx','r')
def parsefile(file):
    gridsize=[]
    origin=[]
    delta=[0,0,0]
    value=[]
    for line in file:
        if line[0]=='#':
            continue
        if line[0:2]=='ob':
            if line[7]=='1':
                # print literal_eval(line[line.find('counts')+7:])
                # break
                gridsize=[int(d) for d in line[line.find('counts') + 7:].split(' ')]
                # print gridsize
                continue
                # break
            if line[7]=='2' or line[7]=='3':
                continue
            #     print delta
            #     break
        if line[0:2]=='or':
            origin=[float(d) for d in line[7:].split(' ')]
            # print origin
            continue
            # break
        if line[0:2] == 'de':
            for i,d in enumerate(line[6:].split(' ')):
                delta[i]+=float(d)
            continue
        if line[0:3]=='att':
            break
        # print line.strip().split(' ')
        value.extend([float(d) for d in line.strip().split(' ')])
        # for v in [float(d) for d in line.strip().split(' ')]:
        #     if v<0:
        #         print 1

    # print len(value)
    gridsize=np.asarray(gridsize)
    origin=np.asarray(origin)
    delta=np.asarray(delta)
    value=np.asarray(value)

    return gridsize,origin,delta,value

def findvalue(dotcloud,gridsize,origin,delta,value):
    ind3d=np.floor((dotcloud-origin)/delta)
    ind1d=ind3d[:,2]+ind3d[:,1]*gridsize[2]+ind3d[:,0]*gridsize[1]*gridsize[2]
    # print value.shape
#     print ind3d
    # print value[ind1d.astype(int)]
    temp=value.reshape(gridsize)
#     print temp[ind3d[:,0].astype(int),ind3d[:,1].astype(int),ind3d[:,2].astype(int)]
    return value[ind1d.astype(int)]

if __name__ == "__main__":

    file = open('/Users/bowendai/Documents/ProteinEng/1A2K_l.pqr.dx', 'r')

    gridsize,origin,delta,value=parsefile(file)
    print origin
    print delta
    # print gridsize
    # print value
    # dc=np.random.rand(5,3)*gridsize*delta+origin
    dc=np.asarray([[37.371,5.541,81.174],[37.371,5.041,81.174],[37.371,4.7261,81.1740000],[37.602,4.7261,81.1740000],[38.102,4.7261,81.1740000],[36.8602,4.7261,81.1740000]])
    print findvalue(dc,gridsize,origin,delta,value)

        # print line
        # print literal_eval(line)
    # print g[5,5,5]