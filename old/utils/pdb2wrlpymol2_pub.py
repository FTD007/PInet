import pymol
from pymol import cmd,stored
import os

folder='your input folder address in which are all pdb files you want to get surface mesh'
sfolder='output folder address'

if not os.path.exists(sfolder):
    os.mkdir(sfolder)
os.chdir(folder)
files=os.listdir(folder)
for f in files:
    cmd.load(folder+f)
    cmd.set('surface_quality', '0')
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    cmd.save(sfolder + f[0:-4]+'.wrl')
    cmd.delete('all')

