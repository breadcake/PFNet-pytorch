# PFNet-pytorch
PyTorch implementation for reproducing Perspective Network (PFNet) results in the paper [Rethinking Planar Homography Estimation Using Perspective Fields](https://eprints.qut.edu.au/126933/) by Rui Zeng, Simon Denman, Sridha Sridharan, Clinton Fookes.
***
### COCO Dataset
- Please refer to [Common Objects in Context](http://cocodataset.org/#home) to download the dataset used in the paper.
***
### Trained Weights.
- Download our trained weights from [provisionally trained weights](https://www.dropbox.com/s/dk29bo0ml6ao7gc/pfnet_0200.h5?dl=0) and put it in the root directory

***
### Dependencies
python 3.6

***
### Training
- Train a PFNet model on the COCO dataset from scratch:
  -  `python train.py --dataset=/path/to/COCO`

***
### Evaluation
- Evaluate the model checkpoint
  - `python evaluate.py --dataset=/path/to/COCO --model=./pfnet_0200.pth`
