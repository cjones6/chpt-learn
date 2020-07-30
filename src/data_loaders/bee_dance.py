import numpy as np
import os
import scipy.io as sio
import torch

from . import create_data_loaders


def get_dataloaders(batch_size=1, data_path='../data/beedance', difference=None, frac_labeled=None, num_labeled=None,
                    num_train=None, num_workers=4, window_size=1):
    """
    Get dataloaders for the Bee Waggle Dance dataset. The dataset we used is the version that can be found here:
    https://github.com/OctoberChang/klcpd_code/tree/master/data/beedance

    :param batch_size: Number of sequences in each batch produced by the dataloaders
    :param data_path: Directory where the data currently exists
    :param difference: How many times to difference the data. The supported values are 0, 1, and 2.
    :param frac_labeled: Fraction of sequences in the training set that should be labeled. Only one of frac_labeled and
                         num_labeled can be specified.
    :param num_labeled: Number of sequences in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param num_train: Number of sequences in the training set. If None, it will be 4.
    :param num_workers: Number of workers to use for the dataloaders
    :param window_size: Size of a sliding window to use on the data. Any integer >= 1 is supported.
    :return: Tuple containing:

            * train_loader: Dataloader for the (labeled+unlabeled) training data
            * train_labeled_loader: Dataloader for the labeled training data.
            * train_unlabeled_loader: Dataloader for the unlabeled training data
            * valid_loader: Dataloader for the validation set
            * test_loader: Dataloader for the test set
    """
    all_xs = []
    all_labels = []
    all_cps = []
    for i in range(1, 7):
        x, labels = load_data(os.path.join(data_path, 'beedance-' + str(i) + '.mat'))
        if difference is not None and difference > 0:
            if difference == 1:
                x = np.diff(x, n=1, axis=0)
                x = np.concatenate((x, [[0, 0, 0]]))
            elif difference == 2:
                x = np.diff(x, n=2, axis=0)
                x = np.concatenate((x, [[0, 0, 0], [0, 0, 0]]))
            else:
                raise NotImplementedError
        if window_size > 1:
            xprev = np.concatenate(([[0, 0, 0]]*(window_size//2), x[:-(window_size//2)]))
            xnext = np.concatenate((x[window_size//2:], [[0, 0, 0]]*(window_size//2)))
            x = np.hstack((xprev, x, xnext))
        all_xs.append(x)
        all_labels.append(labels)
        all_cps.append(np.where(labels==1)[0])

    train_idxs = [0, 1, 2, 3]
    valid_idxs = [4]
    test_idxs = [5]

    if num_train is not None:
        train_idxs = train_idxs[:num_train]

    train_dataset = create_data_loaders.PreloadedDataset([torch.from_numpy(all_xs[idx]) for idx in train_idxs],
                                                         [torch.IntTensor(all_labels[idx]) for idx in train_idxs],
                                                         [torch.IntTensor(all_cps[idx]) for idx in train_idxs],
                                                         transform=None
                                                         )
    valid_dataset = create_data_loaders.PreloadedDataset([torch.from_numpy(all_xs[idx]) for idx in valid_idxs],
                                                         [torch.IntTensor(all_labels[idx]) for idx in valid_idxs],
                                                         [torch.IntTensor(all_cps[idx]) for idx in valid_idxs],
                                                         transform=None
                                                         )
    test_dataset = create_data_loaders.PreloadedDataset([torch.from_numpy(all_xs[idx]) for idx in test_idxs],
                                                         [torch.IntTensor(all_labels[idx]) for idx in test_idxs],
                                                         [torch.IntTensor(all_cps[idx]) for idx in test_idxs],
                                                         transform=None
                                                         )

    return create_data_loaders.generate_dataloaders(train_dataset,
                                                    test_dataset,
                                                    valid_dataset=valid_dataset,
                                                    separate_valid_set=True,
                                                    batch_size=batch_size,
                                                    frac_labeled=frac_labeled,
                                                    num_labeled=num_labeled,
                                                    num_workers=num_workers)


def load_data(data_path):
    """
    Load a .mat file located at data_path.

    :param data_path: Path to the .mat file
    :return: Data and labels contained in the .mat file
    """
    data = sio.loadmat(data_path)
    return data['Y'], data['L']
