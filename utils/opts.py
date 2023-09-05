import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Computing RTs from stable RNNs")

# ========================= Data ==========================
parser.add_argument('dataset_str_train', type=str,
                    help='shorthand for the train set; check ./dataset/dataset_str_mapping')  # required
parser.add_argument('dataset_str_val', type=str,
                    help='shorthand for the train set; check ./dataset/dataset_str_mapping')  # required

parser.add_argument('--data_root', type=str, default="./data", help='directory containing data to load')

parser.add_argument('--subset_train', type=float, default=1.0)

# ========================= Model ==========================
parser.add_argument('--model', type=str, default='hgru',
                    help='which model to run; check ./models/setup_model.py')
parser.add_argument('--timesteps', type=int, default=20,
                    help='number of timesteps for RNNs')
parser.add_argument('--activ', default='softplus',
                    help='activation function inside hgru cell')
parser.add_argument('--kernel_size', type=int, default=15,
                    help='kernel size of hgru')
parser.add_argument('--base_size', type=int, default=150,
                    help='height (=width) to resize images in the case of cocodots')
parser.add_argument('--xavier_gain', type=float, default=1.0,
                    help='gain for xavier init state in hgru')
parser.add_argument('--n_hidden_channels', type=int, default=25,
                    help='number of channels in hidden states')
parser.add_argument('--n_in', type=int, default=4,
                    help='number of input channels')
parser.add_argument('--n_classes', type=int, default=2,
                    help='number of classes')

# ========================= Training algo, loss ==========================
parser.add_argument('--algo', type=str, default="bptt", choices=['bptt', 'rbp'],
                    help='training algo')
parser.add_argument('--penalty', default=False, action='store_true',
                    help='whether to apply the jacobian penalty')
parser.add_argument('--penalty_gamma', type=float, default=1e1,
                    help='importance of the jacobian penalty')
parser.add_argument('--loss_fn', default='cross_entropy', type=str, choices=['cross_entropy', 'EDL'],
                    help='classficiation loss function to use')
parser.add_argument('--annealing_step', default=10.0, type=float,
                    help='annealing_step = args.annealing_step * len(trainloader); annealing_coef = float('
                         'global_step)/annealing_step')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')

parser.add_argument('--adjust_lr', default=False, action='store_true')

# ========================= Monitor Configs ==========================
parser.add_argument('--name', type=str, default='hgru',
                    help='name for the run; also name of the output folder that will be created')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-parallel', '--parallel', default=False, action='store_true',
                    help='Wanna parallelize the training')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--log', default=False, action='store_true')

parser.add_argument('--wandb', default=False, action='store_true')

parser.add_argument('--wandb_project', type=str)

parser.add_argument('--wandb_entity', type=str)

# ========================= Device ==========================
parser.add_argument('--gpu_ids', nargs="+", type=int, default=[],
                    help='which gpus to use. Empty list will be considered as all gpus')
parser.add_argument('--pin_memory', default=False, action='store_true')
