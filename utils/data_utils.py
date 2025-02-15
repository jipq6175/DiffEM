# data utilities



import os, sys, pickle, mrcfile, torch, copy


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go



from tqdm.auto import tqdm
from inspect import isfunction
from torchvision.transforms import ToTensor, Lambda, ToPILImage
from torch.utils.data import DataLoader


# Global Parameters
DIR = '/home/ubuntu/'
DATAPATH = '/home/ubuntu/data/10299_1k.mrcs'
RECONPATH = '/home/ubuntu/reconstructions/'
RECONBIN = '/home/ubuntu/relion/build/bin/relion_reconstruct'
REFINEPATH = '/home/ubuntu/refinements'
REFINEBIN = '/home/ubuntu/relion/build/bin/relion_refine_mpi'
PPBIN = '/home/ubuntu/relion/build/bin/relion_postprocess'


S3MODELURI = 's3://seismictx-cryoem/diffem/trained_models/'
MODELPATH = '/home/ubuntu/trained_models/'

S3VISURI = 's3://seismictx-cryoem/diffem/diffusion_results/'
RLTDIR = '/home/ubuntu/diffusion_results/'



def exists(x): return x is not None

def default(val, d): 
    if exists(val): return val
    return d() if isfunction(d) else d


def s3sync(p1, p2): 
    s3cmd = f'aws s3 sync {p1} {p2}'
    assert os.system(s3cmd) == 0
    return None


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


# for cryo em reconstructions
def get_mrcs_number_and_name(starfile):
    
    assert os.path.isfile(starfile)
    f = open(starfile, 'r')
    last_line = f.readlines()[-2]
    f.close()
    
    s = last_line.split()[3]
    n, name = s.split('@')
    return int(n), name


def reconstruct(starpath, mrcsuri, reconpath=RECONPATH, reconbin=RECONBIN): 

    currdir = os.getcwd()
    n, mrcsname = get_mrcs_number_and_name(starpath)
    mrcss3name = os.path.basename(mrcsuri)

    # get the star file
    os.chdir(reconpath)

    s3cmd = f'aws s3 cp {mrcsuri} ./{mrcsname}'
    assert os.system(s3cmd) == 0

    mrc = mrcfile.open(mrcsname)
    assert mrc.data.shape[0] == n, f'# of images {mrc.data.shape[0]} does not match the star file n = {n}'
    
    relioncmd = f'{reconbin} --i {starpath} --o {mrcss3name[:-5]}_reconstruction.mrc'
    assert os.system(relioncmd) == 0

    os.chdir(currdir)
    
    return os.path.join(reconpath, f'{mrcss3name[:-5]}_reconstruction.mrc') 


# refinement and get the fsc
def refine(starpath, mrcsuri, refpath, model_name, maskpath='/home/ubuntu/refinements/require/30k-mask-job016.mrc', 
           particle='test-5k', particle_diameter=148, lowpass=15, iter=-1, cpus=15, gpu=False, j=10,
           refinepath=REFINEPATH, refinebin=REFINEBIN, ppbin=PPBIN): 
    
    assert os.path.isfile(starpath)
    assert os.path.isfile(refpath)
    assert os.path.isfile(maskpath)

    currdir = os.getcwd()

    n, mrcsname = get_mrcs_number_and_name(starpath)
    mrcss3name = os.path.basename(mrcsuri)
    
    workingdir = os.path.join(refinepath, mrcss3name[:-5])
    os.makedirs(workingdir, exist_ok=True)
    os.chdir(workingdir)

    s3cmd = f'aws s3 cp {mrcsuri} ./{mrcsname}'
    assert os.system(s3cmd) == 0

    mrc = mrcfile.open(mrcsname)
    assert mrc.data.shape[0] == n, f'# of images {mrc.data.shape[0]} does not match the star file n = {n}'
    


    # refinement: expensive
    # relioncmd = f'mpirun -np {cpus} {refinebin} --i {starpath} --ref {refpath} --solvent_mask {maskpath} --particle_diameter {particle_diameter} --lowpass {lowpass} --o {workingdir}/ --j {j} --auto_refine --split_random_halves --iter {iter}'
    '''
    google: mpirun relion_refine_mpi --o Refine3D/run1 --auto_refine --split_random_halves --i Class3D/run1_ct13_it025_data_class1_and3.star --particle_diameter 350 --angpix 1.77 --ref Class3D/run1_ct13_it025_class001.mrc --ini_high 50 --ctf --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale  --j 1 --memory_per_thread 4 --dont_combine_weights_via_disc 
    chris: `which relion_refine_mpi` --o Refine3D/job017/run --auto_refine --split_random_halves --i Import/job001/30k.star --ref Particles/all.mrc --firstiter_cc --ini_high 20 --dont_combine_weights_via_disc --pool 3 --pad 2 --skip_gridding --ctf --particle_diameter 150 --flatten_solvent --zero_mask --solvent_mask MaskCreate/job016/mask.mrc --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale --j 7 --gpu "" --pipeline_control Refine3D/job017/
    '''

    relioncmd = f'mpirun -np {cpus} {refinebin} --i {starpath} --ref {refpath} --solvent_mask {maskpath} --particle_diameter {particle_diameter} --lowpass {lowpass} --o {workingdir}/ --j {j} --auto_refine --split_random_halves --iter {iter} --ini_high 20 --pad 2 --ctf --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --firstiter_cc --low_resol_join_halves 40 --norm --scale'
    if gpu: relioncmd += ' --gpu'
    relioncmd += ' > refinement_run.out'
    print(f'\n{relioncmd}')
    assert os.system(relioncmd) == 0



    # generate the mask
    # relioncmd = '/home/ubuntu/relion/build/bin/relion_mask_create --i _it000_half1_class001.mrc --o mask.mrc --width_soft_edge 10 --extend_inimask 6'
    # print(f'\n{relioncmd}')
    # assert os.system(relioncmd) == 0

    
    # get the max iterations 
    # filenames = [x for x in os.listdir(workingdir) if x.endswith('_half1_class001.mrc')]
    # filenames.sort()
    # it = filenames[-1][3:6]
    # print(f'\nUsing {it} iterations from refinement')

    

    # post-processing
    # relioncmd = f'{ppbin} --i _it{it}_half1_class001.mrc --i2 _it{it}_half2_class001.mrc --auto_bfac false --mask {maskpath} --skip_fsc_weighting'
    # relioncmd = f'{ppbin} --i _it{it}_half1_class001.mrc --i2 _it{it}_half2_class001.mrc --auto_bfac false --mask {maskpath}'
    relioncmd = f'{ppbin} --i _half1_class001_unfil.mrc --i2 _half2_class001_unfil.mrc --auto_bfac false --mask {maskpath} --autob_lowres 10'
    relioncmd += ' > postprocessing_run.out'
    print(relioncmd)
    os.system(relioncmd)

    # sync to s3: 
    s3cmd = f'aws s3 sync {workingdir} s3://seismictx-cryoem/diffem/data/denoised/{particle}/{model_name}/refinement/{mrcss3name[:-5]} --quiet'
    assert os.system(s3cmd) == 0

    os.chdir(currdir)

    return None



    

    

# maybe not bothere with voxel visualization
# def plot_voxel(mrcpath): 

#     assert os.path.isfile(mrcpath)
#     mrc = mrcfile.open(mrcpath)

#     d1, d2, d3 = mrc.data.shape
#     assert d1 == d2 == d3
#     M, m = mrc.data.max(), mrc.data.min()
#     print(M, m)
#     X, Y, Z = np.mgrid[0:d1-1:d1, 0:d1-1:d1, 0:d1-1:d1]

#     fig = go.Figure(data=go.Volume(x=X.flatten(),
#                                    y=Y.flatten(),
#                                    z=Z.flatten(),
#                                    value=mrc.data.flatten(),
#                                    isomin=0.1,
#                                    isomax=0.8,
#                                    opacity=0.5, 
#                                    surface_count=17))
#     fig.update_layout(autosize=False, width=800, height=800)


    # fig.write_html('/home/ubuntu/mrc.html')
    # return None

