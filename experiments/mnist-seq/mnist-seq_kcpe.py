"""
Perform unsupervised kernel-based change-point estimation on the MNIST-seq dataset
"""

import argparse
import numpy as np
import os
import random
import sklearn.metrics
import sys
import time
import torch

sys.path.append('..')
sys.path.append('../..')

from chapydette import cp_estimation

from src.opt import evaluation, opt_structures
from src import default_params as defaults
import src.data_loaders.mnist_seq as mnist

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='Unsupervised kernel-based change-point estimation on the MNIST-seq '
                                             'dataset')
parser.add_argument('--bw', default=None, type=float,
                    help='Bandwidth to use')
parser.add_argument('--data_path', default='../../data/mnist', type=str,
                    help='Location of the MNIST dataset')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPU to use')
parser.add_argument('--min_dist', default=1, type=int,
                    help='Minimum distance between successive change points')
parser.add_argument('--save_path', default='../../results/mnist_kcpe/temp', type=str,
                    help='File path prefix to use to save the model and results')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')

args = parser.parse_args()
print(args)

# Set miscellaneous variables based on the inputs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Create the data loaders
train_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader = \
    mnist.get_dataloaders(data_path=args.data_path, num_workers=0, seed=args.seed)

# Set the bandwidth
if args.bw is None:
    bw = np.median(sklearn.metrics.pairwise.pairwise_distances(torch.cat(valid_loader.dataset.dataset.features)[:1000]
                                                               .numpy().reshape(-1, 28*28)).reshape(-1))
    print('Bandwidth from median heuristic: %0.2f' % bw)
else:
    bw = args.bw

# Set up the path where the results will be saved
save_dir = args.save_path
save_file = os.path.join(save_dir, str(args.bw) + '_' + str(args.min_dist) + '_' + str(args.seed) + '_' +
                         str(time.time()))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Estimate the change points in every sequence in the validation and test sets, evaluate the performance, and save the
# results
hausdorff1s = {'valid_hausdorff1': 0, 'test_hausdorff1': 0}
frobeniuses = {'valid_frobenius': 0, 'test_frobenius': 0}
for dataset_name, data_loader in zip(['valid', 'test'], [valid_loader, test_loader]):
    data_iter = iter(data_loader)
    while True:
        try:
            X, _, _, _, true_cps, _ = next(data_iter)
        except:
            break
        X = X[0].reshape(-1, X[0].shape[1]*X[0].shape[2]).numpy().astype('double')
        true_cps = true_cps[0]
        all_cps, obj = cp_estimation.mkcpe(X, n_cp=len(true_cps), bw=bw,
                                            kernel_type='gaussian-euclidean', min_dist=args.min_dist, return_obj=True)
        hausdorff1s[dataset_name + '_hausdorff1'] += evaluation.compute_hausdorff1(all_cps.flatten(), true_cps)
        frobeniuses[dataset_name + '_frobenius'] += evaluation.compute_frobenius(all_cps.flatten(), true_cps, len(X))
    hausdorff1s[dataset_name + '_hausdorff1'] /= len(data_loader)
    frobeniuses[dataset_name + '_frobenius'] /= len(data_loader)

print('Average Frobenius distance on the test set: %0.2f' % np.mean(frobeniuses['test_frobenius']))
results = opt_structures.Results(save_path=save_file + '_results.pickle')
results.update(0, **hausdorff1s, **frobeniuses)
results.save()
