# applying the denoising model


import os, torch

from utils.data_utils import *
from utils.diff_utils import *

from model.UNet import UNet






if __name__ == '__main__':

    
    T = 1000
    BATCH_SIZE = 400
    IMAGE_SIZE = 128
    CHANNELS = 1

    betas = linear_beta_schedule(T)
    params = get_forward_diffusion_parameters(betas)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    noise2noise = UNet(dim=IMAGE_SIZE, channels=CHANNELS, dim_mults=(1, 2, 4), use_convnext=False)
    noise2noise = torch.nn.DataParallel(noise2noise)
    noise2noise.to(device)

    # load the fully trained model
    noise2noise.load_state_dict(torch.load('/home/ubuntu/trained_models/DiffEM_1000_1000_2022-12-14-19-39-03.pt'))

    
    noise_idx = [500, 750, 900, 949]
    denoise_steps = 50
    hallucinations = [0.0, 0.1, 0.2, 0.5, 1.0]

    # sample images and save as mrcs
    # for dset in ['test-30k', 'test-50k']:
    for dset in ['test-50k']:
        
        test_images = read_mrcs(f'/home/ubuntu/data/{dset}/{dset}.mrcs')
        sample_dct = {T - 1: [], T - 10: [], int(0.9 * T): [], int(0.8 * T): []}
        nepochs = int(test_images.shape[0] / BATCH_SIZE)
        
        deposit_path = os.path.join('/home/ubuntu/data/denoised/', dset)
        os.makedirs(deposit_path, exist_ok=True)

        
        for idx in noise_idx: 

            t = T - 1 - idx

            for noise_scale in hallucinations: 

                denoised_samples = []
                filepath = os.path.join(deposit_path, f'denoised_samples_nl_{t}_halluc_{noise_scale:.1f}_50_steps.mrcs')

                for i in tqdm(range(nepochs), desc=f'-- {dset} idx = {idx}, hall = {noise_scale}'):
                    data = test_images[BATCH_SIZE*i:BATCH_SIZE*(i+1)].clone().to(device)
                    denoised = denoise(noise2noise, data, t, denoise_steps, params, noise_scale=noise_scale)
                    denoised_samples.append(denoised[-1])


                mrcs = np.concatenate(denoised_samples)
                write_mrcs(filepath, mrcs)
                
                s3path = os.path.join('s3://seismictx-cryoem/diffem/data/denoised/', dset, f'denoised_samples_nl_{t}_halluc_{noise_scale:.1f}_50_steps.mrcs')
                s3cmd = f'aws s3 cp {filepath} {s3path}'
                assert os.system(s3cmd) == 0

                rmcmd = f'rm {filepath}'
                assert os.system(rmcmd) == 0