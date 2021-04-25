import sys
import os

folderp='input pdb folder'

foldera='output apbs folder'

if not os.path.exists(foldera):
    os.mkdir(foldera)
pdb2pqr='/path/to/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
apbsflag='--whitespace --ff=amber -v --apbs-input'

apbs='/path/to/apbs-pdb2pqr/bin/apbs'
files=os.listdir(folderp)
os.chdir(foldera)
for f in files:
    if f[-5]=='l':
        continue

    try:
        os.system(pdb2pqr+' '+apbsflag+' '+folderp+f[0:4]+'-l.pdb'+' '+foldera+f[0:4]+'-l.pqr')
    except:
        print 'pqr: '+f

    try:
        os.system(pdb2pqr+' '+apbsflag+' '+folderp+f[0:4]+'-r.pdb'+' '+foldera+f[0:4]+'-r.pqr')
    except:
        print 'pqr: '+f

    try:
        # os.chdir(foldera)
        os.system(apbs+' '+foldera+f[0:4]+'-l.in')
    except:
        print 'abps: '+f

    try:
        # os.chdir(foldera)
        os.system(apbs+' '+foldera+f[0:4]+'-r.in')
    except:
        print 'abps: '+f
