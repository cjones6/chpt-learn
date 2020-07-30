"""
Perform supervised change-point estimation with a KN on the Bee Dance dataset
"""

import argparse
import numpy as np
import os
import random
import sklearn.metrics
import sys
import time
import torch

sys.path.append('../..')

from src import default_params as defaults

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='MLP KN training on the Bee Dance dataset')
parser.add_argument('--batch_size', default=None, type=int,
                    help='Batch size to use in training')
parser.add_argument('--bw', default=0.5, type=float,
                    help='Bandwidth of the kernel')
parser.add_argument('--data_difference', default=1, type=int,
                    help='How much to difference the data')
parser.add_argument('--data_path', default='../../data/beedance', type=str,
                    help='Location of the bee waggle dance dataset')
parser.add_argument('--epsilon', default=-5, type=int,
                    help='log10(epsilon in the first penalty term)')
parser.add_argument('--eval_test_every', default=1, type=int,
                    help='Number of iterations between evaluations of the performance on the test set')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPU to use')
parser.add_argument('--lambda_cov', default=-1, type=int,
                    help='log2(penalty on sqrt(tr(empirical covariance)+penalty_epsilon))')
parser.add_argument('--lr', default=-6, type=int,
                    help='log2(Learning rate)')
parser.add_argument('--min_dist', default=1, type=int,
                    help='Minimum distance between successive change points')
parser.add_argument('--num_filters', default=32, type=int,
                    help='Number of filters per layer in the model')
parser.add_argument('--num_iters', default=100, type=int,
                    help='Number of total iterations to perform')
parser.add_argument('--num_labeled', default=0, type=int,
                    help='Number of sequences that will be labeled')
parser.add_argument('--num_layers', default=5, type=int,
                    help='Number of layers in the network')
parser.add_argument('--num_train', default=4, type=int,
                    help='Number of sequences in the training set')
parser.add_argument('--save_every', default=100, type=int,
                    help='Number of iterations between saves of the model and results')
parser.add_argument('--save_path', default='../results/bee_dance_kn/temp', type=str,
                    help='File path prefix to use to save the model and results')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--window_size', default=3, type=int,
                    help='Window size')

args = parser.parse_args()

# Set miscellaneous variables based on the inputs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from src.opt import opt_structures, train_network
from src.model.ckn import parse_config, net
import src.data_loaders.bee_dance as bee_dance

args.epsilon = 10**args.epsilon
args.lambda_cov = 2**args.lambda_cov
args.lr = 2**args.lr

print(args)

save_dir = args.save_path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_file = save_dir + str(args.batch_size) + '_' + str(args.bw) + '_' + str(args.data_difference) \
            + '_' + str(args.epsilon) + '_' + str(args.lambda_cov) + '_' + str(args.lr) + '_' + str(args.min_dist) \
            + '_' + str(args.num_filters) + '_' + str(args.num_labeled) + '_' + str(args.num_train) \
            + '_' + str(args.num_layers) + '_' + str(args.seed) + '_' + str(time.time()) + '_' + str(args.window_size)

# Create the data loaders
train_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader = \
    bee_dance.get_dataloaders(batch_size=1, data_path=args.data_path, difference=args.data_difference,
                              num_labeled=args.num_labeled, num_train=args.num_train, num_workers=0,
                              window_size=args.window_size)

if args.bw is None:
    bw = np.median(sklearn.metrics.pairwise.pairwise_distances(train_loader.dataset.features[0].numpy()).reshape(-1))
    print('Bandwidth from median heuristic:')
else:
    bw = args.bw

# Load and initialize the model
model_params = parse_config.load_config('../../cfg/kn.cfg')
if args.num_filters > 0:
    for key in model_params:
        model_params[key] *= args.num_layers
    model_params['num_filters'] = [args.num_filters] * args.num_layers
    model_params['patch_sigma'] = [bw] * args.num_layers

layers = parse_config.create_layers(model_params)
model = net.CKN(layers).to(defaults.device)
model.init(train_loader)
print('Done with initialization')

# Set up the data, parameters, model, results, and optimizer objects
data = opt_structures.Data(train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader)
params = opt_structures.Params(batch_size=args.batch_size, ckn=True, epsilon=args.epsilon,
                               eval_test_every=args.eval_test_every, lambda_cov=args.lambda_cov, lambda_params=0,
                               lr=args.lr, lr_schedule=None, min_dist=args.min_dist, num_classes=3,
                               num_iters=args.num_iters, project=True, save_every=args.save_every,
                               save_path=save_file + '_params.pickle', train_w_layers=None)
model = opt_structures.Model(model, save_path=save_file + '_model.pickle')
results = opt_structures.Results(save_path=save_file + '_results.pickle')
optimizer = train_network.TrainSupervised(data, model, params, results)

# Train the model
optimizer.train()
