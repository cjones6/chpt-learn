"""
Perform supervised change-point estimation on simulated data
"""

import argparse
import numpy as np
import os
import random
import sys
import time
import torch

sys.path.append('../..')

import src.default_params as defaults

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='Change-point estimation on simulated data')
parser.add_argument('--epsilon', default=-4, type=int,
                    help='log10(epsilon in the first penalty term)')
parser.add_argument('--eval_test_every', default=1, type=int,
                    help='Number of iterations between evaluations of the performance on the test set')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPU to use')
parser.add_argument('--lambda_cov', default=4, type=int,
                    help='log2(penalty on sqrt(tr(empirical covariance)+penalty_epsilon))')
parser.add_argument('--lambda_params', default=-10, type=int,
                    help="log2(penalty on ||parameters||^2)")
parser.add_argument('--lr', default=-4, type=int,
                    help='log2(Learning rate)')
parser.add_argument('--min_dist', default=1, type=int,
                    help='Minimum distance between successive change points')
parser.add_argument('--num_filters', default=3, type=int,
                    help='Number of filters per layer in the model')
parser.add_argument('--num_iters', default=100, type=int,
                    help='Number of total iterations to perform')
parser.add_argument('--num_labeled', default=0, type=int,
                    help='Number of sequences that will be labeled')
parser.add_argument('--num_train', default=500, type=int,
                    help='Number of sequences in the training set')
parser.add_argument('--save_every', default=100, type=int,
                    help='Number of iterations between saves of the model and results')
parser.add_argument('--save_path', default='../results/simulated/temp', type=str,
                    help='File path prefix to use to save the model and results')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--std', default=-2, type=int,
                    help='log2(Standard deviation of the data)')

args = parser.parse_args()

# Set miscellaneous variables based on the inputs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from src.opt import opt_structures, train_network
from src.data_loaders import simulated
from src.model import sum_sigmoids

args.epsilon = 10**args.epsilon
args.lambda_cov = 2**args.lambda_cov
args.lambda_params = 2**args.lambda_params
args.lr = 2**args.lr
args.std = 2**args.std

print(args)

save_dir = args.save_path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_file = save_dir + str(args.epsilon) + '_' + str(args.lambda_cov) + '_' + str(args.lambda_params) \
            + '_' + str(args.lr) + '_' + str(args.min_dist) + '_' + str(args.num_filters) \
            + '_' + str(args.num_labeled) + '_' + str(args.num_train) + '_' + str(args.seed) + '_' + str(args.std) \
            + '_' + str(time.time())

# Create the data loaders
_, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader = simulated.get_dataloaders(
    ntrain=args.num_train, std=args.std, use_network=True, num_labeled=args.num_labeled, batch_size=args.num_train,
    num_workers=0, seed=args.seed)

# Load and initialize the model
model = sum_sigmoids.SumSigmoids(args.num_filters)
model.apply(sum_sigmoids.init_normal)
model.to(defaults.device)
print('Initial thetas:', torch.stack(list(model.state_dict()['thetas'])))

# Set up the data, parameters, model, results, and optimizer objects
data = opt_structures.Data(train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader)
params = opt_structures.Params(ckn=False, epsilon=args.epsilon, eval_test_every=args.eval_test_every,
                               lambda_cov=args.lambda_cov, lambda_params=args.lambda_params, lr=args.lr,
                               lr_schedule=None, min_dist=args.min_dist, num_classes=2, num_iters=args.num_iters,
                               project=False, save_every=args.save_every, save_path=save_file + '_params.pickle',
                               train_w_layers=None)
model = opt_structures.Model(model, save_path=save_file + '_model.pickle')
results = opt_structures.Results(save_path=save_file + '_results.pickle')
optimizer = train_network.TrainSupervised(data, model, params, results)

# Train the model
optimizer.train()
