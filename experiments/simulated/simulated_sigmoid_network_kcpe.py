"""
Perform unsupervised kernel-based change-point estimation on simulated data
"""

import argparse
import numpy as np
import os
import random
import sklearn.metrics
import sys
import time
import torch

from chapydette import cp_estimation

sys.path.append('..')
sys.path.append('../..')

from src.opt import evaluation, opt_structures
import src.data_loaders.simulated as simulated

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='Change-point estimation on simulated data')
parser.add_argument('--bw', default=None, type=float,
                    help='Bandwidth to use')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPU to use')
parser.add_argument('--kernel', default='gaussian-euclidean', type=str,
                    help='Which kernel to use')
parser.add_argument('--min_dist', default=1, type=int,
                    help='Minimum distance between successive change points')
parser.add_argument('--save_path', default='../../results/simulated_kcpe/temp', type=str,
                    help='File path prefix to use to save the model and results')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--std', default=-2, type=int,
                    help='log2(Standard deviation of the data)')
parser.add_argument('--use_network', default=1, type=int,
                    help='Whether to pass the observations through a network')

args = parser.parse_args()

# Set miscellaneous variables based on the inputs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.std = 2**args.std

print(args)

# Create the data loaders
train_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader = \
    simulated.get_dataloaders(std=args.std, batch_size=1, num_labeled=0, num_workers=0, use_network=args.use_network,
                              seed=args.seed)

# Set the bandwidth
if args.bw is None and args.kernel != 'linear':
    bw = np.median(sklearn.metrics.pairwise.pairwise_distances(train_loader.dataset.features[0].numpy()).reshape(-1))
    print('Bandwidth from median heuristic: %0.2f' % bw)
else:
    bw = args.bw

# Set up the path where the results will be saved
save_dir = args.save_path
save_file = save_dir + str(bw) + '_' + args.kernel + '_' + str(args.min_dist) + '_' + str(args.seed) + '_' + \
            str(args.std) + '_' + str(time.time())

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Estimate the change points in every sequence in the validation and test sets, evaluate the performance, and save the
# results
hausdorff1s = {'train_hausdorff1': 0, 'valid_hausdorff1': 0, 'test_hausdorff1': 0}
frobeniuses = {'train_frobenius': 0, 'valid_frobenius': 0, 'test_frobenius': 0}
for dataset_name, data_loader in zip(['train', 'valid', 'test'], [train_loader, valid_loader, test_loader]):
    for i in range(len(data_loader)):
        X = data_loader.dataset.features[i].cpu().numpy()
        true_cps = data_loader.dataset.true_change_points[i].numpy()
        all_cps, objs = cp_estimation.mkcpe(X, n_cp=len(true_cps), bw=bw, kernel_type=args.kernel,
                                            min_dist=args.min_dist, return_obj=True)
        hausdorff1s[dataset_name + '_hausdorff1'] += evaluation.compute_hausdorff1(all_cps.flatten(), true_cps)
        frobeniuses[dataset_name + '_frobenius'] += evaluation.compute_frobenius(all_cps.flatten(), true_cps, len(X))
    hausdorff1s[dataset_name + '_hausdorff1'] /= len(data_loader)
    frobeniuses[dataset_name + '_frobenius'] /= len(data_loader)

print('Average Frobenius distance on the test set: %0.2f' % np.mean(frobeniuses['test_frobenius']))
results = opt_structures.Results(save_path=save_file + '_results.pickle')
results.update(0, **hausdorff1s, **frobeniuses)
results.save()
