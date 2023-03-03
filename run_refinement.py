# testing the refinement pipeline

import os
from utils.data_utils import refine


if __name__ == '__main__': 

    particle = 'test-30k'
    starpath = '/home/ubuntu/data/test-30k/test-30k.star'
    refpath = '/home/ubuntu/refinements/require/10299-reference.mrc'

    # vanilla model: un-denoised
    model_name = 'vanilla'
    mrcsuri = 's3://seismictx-cryoem/diffem/data/test-30k/30k.mrcs'
    refine(starpath, mrcsuri, refpath, model_name, gpu=True, cpus=3, particle=particle, lowpass=15)

    # testing models: denoised
    model_names = ['DiffEM_1000_1000_2022-12-14-19-39-03', 'DiffEM_2023-02-06-21-50-46']
    for model_name in model_names: 
        for nl in [50, 99, 249, 499]: 
            for hall in [0.0, 0.1, 0.2, 0.5, 1.0]: 
                mrcsuri = f's3://seismictx-cryoem/diffem/data/denoised/test-30k/{model_name}/denoised_samples_nl_{nl}_halluc_{hall:.1f}_50_steps.mrcs'
                refine(starpath, mrcsuri, refpath, model_name, gpu=True, cpus=3, particle=particle, lowpass=15)
                assert os.system(f'rm -r /home/ubuntu/refinements/denoised_samples_nl_{nl}_halluc_{hall:.1f}_50_steps') == 0
    
    print('SUCCESS!!')