import os, torch, random

# from tqdm.auto import tqdm
from datetime import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils.parser import parser
from utils.data_utils import *
from utils.wandb_utils import *
from utils.diff_utils import visualize_samples, get_variance_scheduling
from utils.training_utils import train_diffusion, deposit_model

from model.UNet import UNet
from utils.diff_utils import linear_beta_schedule, get_forward_diffusion_parameters



DIR = '/home/ubuntu/'
RLTDIR = '/home/ubuntu/diffusion_results/'



if __name__ == '__main__':

    # argparse
    args = parser.parse_args()

    # seed, experiment and architecture
    seed, experiment, architecture = args.seed, args.experiment, args.architecture
    if seed is not None: random.seed(seed)

    # experiment name
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f'{experiment}_{now}'
    
    # diffusion parameters
    diffusion_parameters = dict(diffusion_steps=args.diffusion_steps, 
                                scheduling=args.var_schedule)
    params = get_variance_scheduling(args.diffusion_steps, args.var_schedule)

    # training parameters
    datapath = DATAPATH if args.data_path is None else args.data_path
    training_parameters = dict(datapath=datapath, 
                               batch_size=args.batch_size, 
                               nepochs=args.epochs,
                               learning_rate=args.learning_rate, 
                               weight_decay=args.weight_decay,
                               device='cuda:0' if torch.cuda.is_available() else 'cpu', 
                               optimizer=args.optimizer, 
                               loss_type=args.loss_type)

    # model parameters
    if args.dim_mults == 2: dim_mults = (1, 2)
    elif args.dim_mults == 4: dim_mults = (1, 2, 4)
    else: dim_mults = (1, 2, 4, 8)

    model_parameters = dict(dim=args.image_size, 
                            channels=args.channels, 
                            dim_mults=dim_mults, 
                            resnet_block_groups=args.resnet_block_groups,
                            use_convnext=args.use_convnext)


    # dataloader
    dataloader = get_dataloader(training_parameters['datapath'], batch_size=training_parameters['batch_size'], shuffle=True)
    

    # wandb setup before training
    wandb.init(project='DiffEMv1', name=experiment_name, reinit=True, dir=DIR, 
               config=dict(**diffusion_parameters, **model_parameters, **training_parameters))

    print('-- Data loader size = ', len(dataloader))

    noise2noise = train_diffusion(dataloader, training_parameters, params, model_parameters)

    # save model and deposit to s3
    if args.keep_model: deposit_model(noise2noise, params, experiment_name)
    
    # -- testing ... 
    # reload the model
    # experiment_name = 'DiffEM_2023-02-02-23-08-44'
    # noise2noise = UNet(**model_parameters)
    # noise2noise = torch.nn.DataParallel(noise2noise)
    # noise2noise.to(training_parameters['device'])
    # noise2noise.load_state_dict(torch.load(f'/home/ubuntu/trained_models/{experiment_name}.pt'))
    # params = get_forward_diffusion_parameters(linear_beta_schedule(1000))


    # generate images for multiple images
    print(f'-- Sampling from trained model: {experiment_name} ... ')
    rlt_folder = os.path.join(RLTDIR, experiment_name)
    rlt_uri = os.path.join(S3VISURI, experiment_name)
    os.makedirs(rlt_folder, exist_ok=True)
    sample_path = os.path.join(rlt_folder, f'{experiment_name}_random_samples.png')
    samples = visualize_samples(noise2noise, args.image_size, 15, args.channels, params, sample_path)
    wandb_summary(samples=wandb.Image(sample_path))
    s3sync(rlt_folder, rlt_uri)


    wandb.finish()
    print('SUCCESS!')


    

    



                        
