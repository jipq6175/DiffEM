# diffusion utilities


import torch
import numpy as np

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from .data_utils import exists, plot



# linear beta scheduling
def linear_beta_schedule(timesteps): 
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


# quadratic beta scheduling
def quadratic_beta_schedule(timesteps):
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


# sigmoid beta scheduling
def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# Cosine beta scheduling 
# Might be the optimal one
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# Getting the forward diffusion parameters
def get_forward_diffusion_parameters(betas): 

    T = betas.shape[0]
    
    params = dict()
    params['betas'] = betas
    params['alphas'] = 1.0 - betas
    params['alphas_cumprod'] = torch.cumprod(params['alphas'], axis=0)
    params['alphas_cumprod_prev'] = torch.nn.functional.pad(params['alphas_cumprod'][:-1], (1, 0), value=1.0)
    params['sqrt_recip_alphas'] = torch.sqrt(1.0 / params['alphas'])

    # diffusion
    params['sqrt_alphas_cumprod'] = torch.sqrt(params['alphas_cumprod'])
    params['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1.0 - params['alphas_cumprod'])

    # posterior
    params['posterior_variance'] = params['betas'] * (1.0 - params['alphas_cumprod_prev']) / (1.0 - params['alphas_cumprod'])

    return params


# getting the preset value in the sampling stage
def extract(a, t, x_shape): 
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion process
def q_sample(x0, t, params, noise=None): 

    if not exists(noise): noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = extract(params['sqrt_alphas_cumprod'], t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(params['sqrt_one_minus_alphas_cumprod'], t, x0.shape)
    
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x0, t, params): 
    x_noisy = q_sample(x0, t, params)
    return x_noisy


# get the diffusion params
def get_variance_scheduling(T, scheduling):
    print(f'-- Using {scheduling} variance scheduling.')
    assert scheduling in ['cosine', 'linear', 'sigmoid', 'quadratic'], f'Wrong scheduling = {scheduling}..'
    if scheduling == 'cosine': betas = cosine_beta_schedule(T)
    elif scheduling == 'linear': betas = linear_beta_schedule(T)
    elif scheduling == 'sigmoid': betas = sigmoid_beta_schedule(T)
    else: betas = quadratic_beta_schedule(T)
    return get_forward_diffusion_parameters(betas)
    

# Sampling: Samples from the noisy gaussian 
# stochastic or naively deterministic 
# noise_scale = 1.0: stochastic
# noise_level = 0.0: deterministic
# noise_level = 1/n: if we average n^2 denoised samples together 
# noise_level is the degree of hallucination
@torch.no_grad()
def p_sample(model, x, t, params, t_index, noise_scale=1.0): 
    
    assert noise_scale >= 0.0 and noise_scale <= 1.0

    betas_t = extract(params['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(params['sqrt_one_minus_alphas_cumprod'], t, x.shape)
    sqrt_recip_alphas_t = extract(params['sqrt_recip_alphas'], t, x.shape)
    
    # Equation 11 in Ho et al. 2020
    # Use the noise-2-noise model to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0: return model_mean
    else: 
        posterior_variance_t = extract(params['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise * noise_scale


@torch.no_grad()
def p_sample_loop(model, shape, params, T): 
    
    device = next(model.parameters()).device
    
    b = shape[0]
    
    # start with pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, T)), desc='-- Sampling Loop Timestep', total=T):
        img = p_sample(model, img, torch.full((b, ), i, device=device, dtype=torch.long), params, i, noise_scale=1.0)
        imgs.append(img.cpu().numpy())
    
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size, channels, params): 
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), params=params, T=len(params['betas']))


# Denoising: 
# Treat xt as the noise level at t = t
# and then denoise for nt time steps
@torch.no_grad()
def p_denoising_loop(model, xt, t, nt, params, noise_scale=1.0): 
    
    if nt == 0: return xt

    t_final = t - nt
    T = len(params['betas'])
    
    assert t_final < T and t_final >= 0
    
    device = next(model.parameters()).device
    
    shape = xt.shape
    b = shape[0]
    
    # start with pure noise (for each example in the batch)
    img = xt.clone()
    imgs = []
    imgs.append(img.cpu().numpy())
    
    for i in tqdm(reversed(range(t_final, t)), desc=f'-- Denoising {t} to {t_final}'):
        img = p_sample(model, img, torch.full((b, ), i, device=device, dtype=torch.long), params, i, noise_scale=noise_scale)
        imgs.append(img.cpu().numpy())
    
    return imgs

@torch.no_grad()
def denoise(model, xt, t, nt, params, noise_scale=1.0): return p_denoising_loop(model, xt, t, nt, params=params, noise_scale=noise_scale)




# diffusion Losses
def p_losses(denoising_model, x0, t, params, noise=None, loss_type='L1'): 
    
    assert loss_type in ['L1', 'L2', 'huber']
    if not exists(noise): noise = torch.randn_like(x0)
    
    x_noisy = q_sample(x0, t, params, noise=noise)
    predicted_noise = denoising_model(x_noisy, t)
    
    if loss_type == 'L1': loss = torch.nn.functional.l1_loss(noise, predicted_noise)
    elif loss_type == 'L2': loss = torch.nn.functional.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber': loss = torch.nn.functional.smooth_l1_loss(noise, predicted_noise)
    else: raise NotImplementedError()
    
    return loss


# sample the images and deposit 
def visualize_samples(noise2noise, image_size, batch_size, channels, params, savepath):
    samples = sample(noise2noise, image_size=image_size, batch_size=batch_size, channels=channels, params=params)
    T = len(params['betas'])
    ts = [0, 50, int(0.5 * T), int(0.75 * T), int(0.9 * T), T - 10, T - 5, T - 3, T - 2, T - 1]
    samples_for_plot = [[torch.from_numpy(samples[j][i, 0, :, :]) for j in ts] for i in range(batch_size)]
    plot(samples_for_plot)
    plt.savefig(savepath)
    return samples


