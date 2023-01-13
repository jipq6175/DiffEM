# data utilities



import os, sys, pickle, mrcfile, torch, copy


import numpy as np
import matplotlib.pyplot as plt



from tqdm.auto import tqdm
from inspect import isfunction
from torchvision.transforms import ToTensor, Lambda, ToPILImage
from torch.utils.data import DataLoader


# Global Parameters
DATAPATH = '/home/ubuntu/data/10299_1k.mrcs'



def exists(x): return x is not None

def default(val, d): 
    if exists(val): return val
    return d() if isfunction(d) else d



# read the mrcs and get the stacked images as numpy 
def read_mrcs(datapath=DATAPATH): 
    
    assert os.path.isfile(datapath)
    mrc = mrcfile.open(datapath)
    assert mrc.is_image_stack()
    
    M, m = mrc.data.max(), mrc.data.min()
    print(f'pixel (max, min) = ({M:.2f}, {m:.2f})')
    
    return torch.tensor(copy.deepcopy(mrc.data)).unsqueeze(1)

# get the dataloader 
def get_dataloader(datapath=DATAPATH, batch_size=32, shuffle=True): 
    return DataLoader(read_mrcs(datapath), batch_size=batch_size, shuffle=shuffle)


# write the numpy array to mrcs
def write_mrcs(datapath, array): 

    dims = array.shape

    if len(dims) == 4: 
        assert dims[1] == 1
        mrcfile.write(datapath, array[:, 0, :, :])
    elif len(dims) == 3: mrcfile.write(datapath, array)
    else: raise NotImplementedError()

    return None



# plotting the images from a list
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs): 
    
    # make 2d even if there is just 1 row
    if not isinstance(imgs[0], list): imgs = [imgs]
    
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    
    fig, axes = plt.subplots(figsize=(20 * num_cols, 20 * num_rows), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs): 
        # row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row): 
            ax = axes[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    if with_orig: 
        axes[0, 0].set(title='Original Image')
        axes[0, 0].title.set_size(8)
    
    if exists(row_title): 
        for row_idx in range(num_rows): 
            axes[row_idx, 0].set(ylabel=row_title[row_idx])
    
    plt.tight_layout()