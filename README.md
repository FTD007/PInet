# PInet
## Setup
run 
> python setup.py

## Train
run
> python uitls/train_richdbd2_fixed6mmgk.py -dataset [your dataset]

## Load pretrained model
In python with torch imported
> classifier.load_state_dict(torch.load('model/seg_model_protein_15.pth'))
