import argparse

parser = argparse.ArgumentParser(description="DiffEM Training")


# Model Parameters
parser.add_argument("--architecture", type=str, help="Noise2Noise Architecture", choices=['UNet'], default='UNet', required=False)

parser.add_argument("--image_size", type=int, default=128, help="Image Size")
parser.add_argument("--channels", type=int, default=1, help="Number of Channels")
parser.add_argument("--dim_mults", type=int, choices=[2, 4, 8], default=4, help="Dimension Multiplication for the UNet")
# parser.add_argument("--dropout", type=float, default=0.25, help="Dropout (for GN)")
parser.add_argument("--resnet_block_groups", type=int, default=8)
parser.add_argument("--use_convnext", type=bool, default=False, help="Use ConvNeXT core for UNet?")
parser.add_argument("--experiment", type=str, default='DiffEM', help="Prefix for experiment")



# Data parameters
parser.add_argument("--data_path", type=str, default=None, help="Path to mrcs training images")
# parser.add_argument("--testing_size", type=float, default=0.2, help="Testing size")
# parser.add_argument("--splitting", type=str, default='seq', choices=['seq', 'naive'], help='Data splitting method')



# Training parameters
parser.add_argument("--batch_size", type=int, default=32, help="Minibatch Size")
# parser.add_argument("--n_sampling", type=int, default=10, help="Number of balance sampling per batch")
parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam', 'SGD', 'AdamW'], help="Optimizer")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
# parser.add_argument("--log_every", type=int, default=50, help="Print log every")
parser.add_argument("--loss_type", type=str, default='L1', choices=['L1', 'L2'], help='L1 or L2 loss')


# Diffusion Parameters
parser.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion Steps")
parser.add_argument("--var_schedule", type=str, choices=['cosine', 'linear', 'sigmoid', 'quadratic'], default='linear')



# Other parameters
# parser.add_argument("--gpu", type=int, default=0, help="GPU index")
# parser.add_argument("--log", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=99, help="Random seed")
parser.add_argument("--keep_model", type=bool, default=True, help="Save model")
# parser.add_argument("--visualization", default=False, action='store_true')