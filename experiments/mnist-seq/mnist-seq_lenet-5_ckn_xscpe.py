"""
Perform supervised change-point estimation with the LeNet-5 CKN on the MNIST-seq dataset
"""

import argparse
import numpy as np
import os
import random
import sys
import time
import torch

sys.path.append('..')
sys.path.append('../..')

from src import default_params as defaults
import src.data_loaders.mnist_seq as mnist
from src.model.ckn import parse_config, net
from src.opt import opt_structures, train_network

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='Supervised change-point estimation with the LeNet-5 CKN on the MNIST-seq'
                                             'dataset')
parser.add_argument('--batch_size', default=50, type=int,
                    help='Batch size to use in training')
parser.add_argument('--data_path', default='../../data/mnist', type=str,
                    help='Location of the MNIST dataset')
parser.add_argument('--epsilon', default=-3, type=int,
                    help='log10(epsilon in the first penalty term)')
parser.add_argument('--eval_test_every', default=1, type=int,
                    help='Number of iterations between evaluations of the performance on the test set')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPU to use')
parser.add_argument('--lambda_cov', default=3, type=int,
                    help='log2(penalty on sqrt(tr(empirical covariance)+penalty_epsilon))')
parser.add_argument('--lr', default=-3, type=int,
                    help='log2(Learning rate)')
parser.add_argument('--min_dist', default=1, type=int,
                    help='Minimum distance between successive change points')
parser.add_argument('--num_filters', default=32, type=int,
                    help='Number of filters per layer in the model')
parser.add_argument('--num_iters', default=100, type=int,
                    help='Number of total iterations to perform')
parser.add_argument('--num_labeled', default=0, type=int,
                    help='Number of sequences that will be labeled')
parser.add_argument('--num_train', default=500, type=int,
                    help='Number of sequences in the training set')
parser.add_argument('--save_every', default=100, type=int,
                    help='Number of iterations between saves of the model and results')
parser.add_argument('--save_path', default='../results/mnist_lenet5/temp', type=str,
                    help='File path prefix to use to save the model and results')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')

args = parser.parse_args()

# Set miscellaneous variables based on the inputs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.epsilon = 10**args.epsilon
args.lambda_cov = 2**args.lambda_cov
args.lr = 2**args.lr
bw = 0.7

print(args)

save_dir = args.save_path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_file = save_dir + str(args.batch_size) + '_' + str(bw) + '_' + str(args.epsilon) + '_' + str(args.lambda_cov) \
            + '_' + str(args.lr) + '_' + str(args.min_dist) + '_' + str(args.num_filters) \
            + '_' + str(args.num_labeled) + '_' + str(args.num_train) + '_' + str(args.seed) + '_' + str(time.time())

# Create the data loaders
train_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader = \
    mnist.get_dataloaders(data_path=args.data_path, num_labeled=args.num_labeled, num_train=args.num_train,
                          num_workers=0, seed=args.seed)
print('Size of training set:', len(train_loader.dataset))

# Load and initialize the model
model_params = parse_config.load_config('../../cfg/lenet-5_ckn.cfg')
if args.num_filters > 0:
    nlayers = len(model_params['num_filters'])
    model_params['num_filters'] = [args.num_filters] * nlayers
    model_params['patch_sigma'] = [bw] * nlayers

layers = parse_config.create_layers(model_params)
model = net.CKN(layers).to(defaults.device)
model.init(train_loader)
print('Done with initialization')

# Set up the data, parameters, model, results, and optimizer objects
data = opt_structures.Data(train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader)
params = opt_structures.Params(batch_size=args.batch_size, ckn=True, epsilon=args.epsilon,
                               eval_test_every=args.eval_test_every, lambda_cov=args.lambda_cov, lambda_params=0,
                               lr=args.lr, lr_schedule=None, min_dist=args.min_dist, num_classes=10,
                               num_iters=args.num_iters, project=True, save_every=args.save_every,
                               save_path=save_file + '_params.pickle', train_w_layers=[0, 2, 4, 5])
model = opt_structures.Model(model, save_path=save_file + '_model.pickle')
results = opt_structures.Results(save_path=save_file + '_results.pickle')
optimizer = train_network.TrainSupervised(data, model, params, results)

# Train the model
optimizer.train()
