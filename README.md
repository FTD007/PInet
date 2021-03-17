# PInet
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
dbd
dataset folder should follow dbd folder structure
input data should be a n-by-5 matrix, with columns' order [x,y,z,electrostatic,hydrophobicity]
