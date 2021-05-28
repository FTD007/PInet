# PInet
## Simple Preprocess  
run  

> cd utils  

Just for testing  

> python PreProcessLifeSavor.py 2I25-l.pdb 2I25-r.pdb  

Generate label for trainning  

> python PreProcessLifeSavor.py 2I25-l.pdb 2I25-r.pdb train  

## Setup
run 
> pip install -e .

## Train
run
> python utils/train_richdbd2_fixed6mmgk.py -dataset [your dataset] \
for example here \
> python utils/train_richdbd2_fixed6mmgk.py -dataset dbd


## Load pretrained model
In python with torch imported
> classifier.load_state_dict(torch.load('model/seg_model_protein_15.pth'))

## Sample dataset folder
Dbd 5 
dataset folder should follow dbd folder structure
input data should be a n-by-5 matrix, with columns' order [x,y,z,electrostatic,hydrophobicity]   
dbd data are now avalable at https://www.dropbox.com/sh/qqi9op061mfxbmo/AADibYuDdMF4n2bDS3uqiEVha?dl=0
  
## Preprocess Helper
utils/pdb2wrlpymol2_pub.py  
change pdb files folder in the script and for a given complex ABCD.pdb, split the ligand and receptor as ABCD_l.pdb and ABCD_r.pdb.  
  
utils/pdb2pqrall_pub.py  
compute apbs for all pdb files. Need to change path in script for your pdb2pqr and apbs binary exe file.  

utils/transdata2.m  
set your path to data folder which will be your train script --dataset input. It will create coordinate data and interface label file.

utils/RichFeatureApbsCon_pub.py  
modify matlab output 3 dims coordinate feature data file to 5 dims coordinate+electrostatic+hydrophobicity feature.  

