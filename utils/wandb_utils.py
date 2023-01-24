# utilities for weights and biases

import wandb

HOST = 'https://seismic.wandb.io/'
wandb.login(host=HOST)


# wandb logging 
def wandb_logging(**kwargs): 
    wandb.log(kwargs)
    return None

# wandb summary
def wandb_summary(**kwargs):
    if kwargs: 
        for k, v in kwargs.items(): 
            wandb.summary[k] = v
    return None


def fetch_run(run_path): 
    run = wandb.Api().run(run_path)
    return run

