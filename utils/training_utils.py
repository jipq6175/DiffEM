# training cryoEM diffusion/denoising model

import os, torch

import numpy as np

from datetime import datetime
from tqdm.auto import tqdm

from torch.optim import Adam, AdamW, SGD

from utils.data_utils import S3MODELURI, MODELPATH, plot
from utils.diff_utils import *
from utils.wandb_utils import *
from model.UNet import *




def train_diffusion(dataloader, training_parameters, diffusion_parameters, model_parameters): 

    T = diffusion_parameters['diffusion_steps']
    scheduling = diffusion_parameters['scheduling']
    assert scheduling in ['cosine', 'linear', 'sigmoid', 'quadratic'], f'Wrong scheduling = {scheduling}..'

    device = torch.device(training_parameters['device'])
    optimizer = training_parameters['optimizer']
    assert optimizer in ['Adam', 'AdamW', 'SGD']
    learning_rate, weight_decay = training_parameters['learning_rate'], training_parameters['weight_decay']
    
    nepochs = training_parameters['nepochs']
    print(f'-- Using {device} and {optimizer} for training.')
    loss_type = training_parameters['loss_type']

    # diffusion stuff
    if scheduling == 'cosine': betas = cosine_beta_schedule(T)
    elif scheduling == 'linear': betas = linear_beta_schedule(T)
    elif scheduling == 'sigmoid': betas = sigmoid_beta_schedule(T)
    else: betas = quadratic_beta_schedule(T)
    params = get_forward_diffusion_parameters(betas)
    
    # models
    noise2noise = UNet(**model_parameters)
    print('-- Using', torch.cuda.device_count(), 'GPUs!')
    noise2noise = torch.nn.DataParallel(noise2noise)
    noise2noise.to(device)
    print('-- Number of Parameters = ', sum(p.numel() for p in noise2noise.parameters() if p.requires_grad))

    

    if optimizer == 'AdamW': opt = AdamW(noise2noise.parameters(), lr=learning_rate)
    elif optimizer == 'SGD': opt = SGD(noise2noise.parameters(), lr=learning_rate)
    else: opt = Adam(noise2noise.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    for epoch in tqdm(range(nepochs + 1), desc='-- Training DiffEM Model...'): 

        losses = []

        for step, batch_images in enumerate(dataloader): 

            opt.zero_grad()

            batch_size = batch_images.shape[0]
            batch_images = batch_images.to(device)

            # Algorithm 1, line 3
            t = torch.randint(0, T, (batch_size, ), device=device, dtype=torch.long)
            loss = p_losses(noise2noise, batch_images, t, params, loss_type=loss_type)

            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        # log the loss
        wandb_logging(loss=np.array(losses).mean())
    
    # summary the final loss
    wandb_summary(loss=np.array(losses).mean())
    

    return noise2noise, params
    


# save and upload the model
def deposit_model(noise2noise, model_name, model_path=MODELPATH, s3_uri=S3MODELURI): 
    
    model_file = os.path.join(model_path, f'{model_name}.pt')
    torch.save(noise2noise.state_dict(), model_file)
    
    s3_file = f'{s3_uri}{model_name}.pt'
    s3cmd = f'aws s3 cp {model_file} {s3_file}'
    assert os.system(s3cmd) == 0

    return None


