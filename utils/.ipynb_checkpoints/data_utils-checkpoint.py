# data utilities



import os, sys, pickle, mrcfile, torch, copy


import numpy as np
import matplotlib.pyplot as plt



from tqdm.auto import tqdm
from torchvision.transforms import ToTensor, Lambda, ToPILImage
from torch.utils.data import DataLoader


# Global Parameters
DATAPATH = '/home/ubuntu/data/10299_1k.mrcs'



# read the mrcs and get the stacked images as numpy 
def read_mrcs(datapath=DATAPATH): 
    
    assert os.path.isfile(datapath)
    mrc = mrcfile.open(datapath)
    assert mrc.is_image_stack()
    
    M, m = mrc.data.max(), mrc.data.min()
    print(f'pixel (max, min) = ({M:.2f}, {m:.2f})')
    
    return torch.tensor(copy.deepcopy(mrc.data)).unsqueeze(1)


def get_dataloader(datapath=DATAPATH, batch_size=32, shuffle=True): 
    return DataLoader(read_mrcs(datapath), batch_size=batch_size, shuffle=shuffle)