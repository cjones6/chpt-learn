import numpy as np
import torch

from . import create_data_loaders


def get_dataloaders(ntrain=500, nvalid=100, ntest=100, std=-2, batch_size=1, batch_size_labeled=None,
                    batch_size_unlabeled=None, frac_labeled=None, max_cp=10, min_dist=5, num_labeled=None,
                    num_workers=4, seed=None, sequence_length=100, use_network=False):
    """
    Create dataloaders that load sequences of simulated data that will be used for change-point estimation.

    :param ntrain: Number of sequences in the training set
    :param nvalid: Number of sequences in the validation set
    :param ntest: Number of sequences in the test set
    :param std: Standard deviation of the normal distribution each observation is drawn from
    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param frac_labeled: Fraction of sequences in the training set that should be labeled. Only one of frac_labeled and
                         num_labeled can be specified.
    :param max_cp: Maximum number of change points in a sequence
    :param min_dist: Minimum allowable distance between change points
    :param num_labeled: Number of sequences in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param num_workers: Number of workers to use for the dataloaders
    :param seed: Seed to use when creating the training and test sequences
    :param sequence_length: Length of each sequence of digits
    :param use_network: Whether to transform the data by feeding it through an RBF network
    :return: Tuple containing:

            * train_loader: Dataloader for the (labeled+unlabeled) training data
            * train_labeled_loader: Dataloader for the labeled training data. If stratified_sampling=True, each batch
                                    has equal representation from each class.
            * train_unlabeled_loader: Dataloader for the unlabeled training data
            * valid_loader: Dataloader for the validation set
            * test_loader: Dataloader for the test set
    """
    train_sequences, train_sequences_labels, train_sequences_cps = generate_sequences(ntrain, sequence_length,
                                                                                      max_cp, min_dist, std, seed)
    valid_sequences, valid_sequences_labels, valid_sequences_cps = generate_sequences(nvalid, sequence_length,
                                                                                      max_cp, min_dist, std, seed+1)
    test_sequences, test_sequences_labels, test_sequences_cps = generate_sequences(ntest, sequence_length,
                                                                                   max_cp, min_dist, std, seed+2)

    if use_network:
        train_sequences, valid_sequences, test_sequences = network(train_sequences, valid_sequences, test_sequences,
                                                                   seed=seed)


    train_dataset = create_data_loaders.PreloadedDataset(train_sequences,
                                                         train_sequences_labels,
                                                         train_sequences_cps)
    valid_dataset = create_data_loaders.PreloadedDataset(valid_sequences,
                                                         valid_sequences_labels,
                                                         valid_sequences_cps)
    test_dataset = create_data_loaders.PreloadedDataset(test_sequences,
                                                        test_sequences_labels,
                                                        test_sequences_cps)

    return create_data_loaders.generate_dataloaders(train_dataset,
                                                    test_dataset,
                                                    valid_dataset=valid_dataset,
                                                    separate_valid_set=True,
                                                    batch_size=batch_size,
                                                    batch_size_labeled=batch_size_labeled,
                                                    batch_size_unlabeled=batch_size_unlabeled,
                                                    frac_labeled=frac_labeled,
                                                    num_labeled=num_labeled,
                                                    num_workers=num_workers)


def select_cp_locations(n_cp, sequence_length, min_dist=5):
    """
    Given a number of change points and a sequence length, randomly choose the change point locations, subject to the
    constraint that the minimum distance between change points must be at least min_dist.

    :param n_cp: Number of change points to choose
    :param sequence_length: Length of the sequence from which change points are being chosen
    :param min_dist: Minimum allowable distance between change points
    :return: cps: The index of the LAST ENTRY in each segment
    """
    cps = -1*np.ones(n_cp, dtype=int)
    cp_idx = 0
    possible_cps = range(min_dist-1, sequence_length-min_dist)
    while cp_idx < n_cp:
        possible_cp = np.random.choice(possible_cps)
        if np.min(np.abs(cps-possible_cp)) >= min_dist:
            cps[cp_idx] = possible_cp
            cp_idx += 1

    return sorted(cps)


def generate_sequences(n, sequence_length, max_cp, min_dist, std, seed):
    """
    Generate sequences of inputs in which successive entries are usually from the same class.

    :param n: Number of sequences
    :param sequence_length: Length of desired sequences
    :param max_cp: Maximum number of desired change points per sequence
    :param min_dist: Minimum distance between change points
    :param std: Standard deviation of the normal distribution each observation is drawn from
    :param seed: Random seed to use
    :return: Tuple containing:

            * sequences: Sequences of inputs of length sequence_length
            * sequences_labels: Sequences of labels corresponding to the above input sequences
            * sequences_cps: List of change points in the labels in the above sequences
    """
    np.random.seed(seed)
    sequences = []
    sequences_labels = []
    sequences_cps = []
    mean = 1

    for i in range(n):
        # Choose the number of change points
        n_cp = np.random.choice(range(1, max_cp + 1))
        # Select the locations of those change points
        cps = select_cp_locations(n_cp, sequence_length, min_dist=min_dist)
        cps = [-1] + cps + [sequence_length - 1]
        # Generate data for each segment
        sequence = torch.zeros((sequence_length, 1))
        sequence_labels = torch.zeros(sequence_length, dtype=torch.int32)
        for j in range(len(cps)-1):
            sequence[cps[j]+1:cps[j+1]+1] = torch.normal(mean=mean, std=std, size=(cps[j+1]-cps[j], 1))
            sequence_labels[cps[j]+1:cps[j+1]+1] = torch.ones(cps[j+1]-cps[j])*mean
            mean = (mean != 1)
        cps = cps[1:-1]

        # Check the resultant sequences
        if len(cps) > 0:
            assert np.min(np.abs(np.diff(np.asarray(sequence_labels))[cps])) > 0
            assert np.max(np.abs(np.diff(np.asarray(sequence_labels))
                                 [list(set(range(len(sequence_labels)-1)).difference(cps))])) == 0
        else:
            assert np.max(np.abs(np.diff(np.asarray(sequence_labels)))) == 0
        if len(cps) > 1:
            assert np.min(np.diff(cps)) >= 0
        assert len(sequence_labels) == sequence_length
        assert len(sequence) == sequence_length

        # Store the sequence if it has at least one change point (note that the number of change points can change in
        # the function select_labels if edge cases are encountered)
        if len(cps) > 0:
            sequences.append(sequence)
            sequences_labels.append(torch.IntTensor(sequence_labels))
            sequences_cps.append(torch.IntTensor(cps))

    return sequences, sequences_labels, sequences_cps


def network(train_sequences, valid_sequences, test_sequences, m=3, seed=None):
    """
    Transform the input sequences by feeding them through a two-layer RBF network that is a sum of m sigmoids.

    :param train_sequences: List of training sequences to be transformed
    :param valid_sequences: List of validation sequences to be transformed
    :param test_sequences: List of test sequences to be transformed
    :param m: Number of sigmoids used in the RBF network
    :param seed: Random seed to use when drawing the parameters of the RBF network.
    :return: Tuple containing:

        * train_sequences: Transformed training sequences
        * valid_sequences: Transformed validation sequences
        * test_sequences: Transformed test sequences
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    thetas = torch.normal(mean=0, std=3, size=(m, 3))
    print('True thetas:', thetas)
    for i in range(len(train_sequences)):
        train_sequences[i] = torch.sum(thetas[:, 2]*torch.exp(-0.5*(torch.mm(train_sequences[i],
                                        thetas[:, 0].unsqueeze(0)) + thetas[:, 1]) ** 2), 1, keepdim=True)
    for i in range(len(valid_sequences)):
        valid_sequences[i] = torch.sum(thetas[:, 2] * torch.exp(-0.5 * (torch.mm(valid_sequences[i],
                                        thetas[:, 0].unsqueeze(0)) + thetas[:, 1]) ** 2), 1, keepdim=True)
    for i in range(len(test_sequences)):
        test_sequences[i] = torch.sum(thetas[:, 2] * torch.exp(-0.5 * (torch.mm(test_sequences[i],
                                        thetas[:, 0].unsqueeze(0)) + thetas[:, 1]) ** 2), 1, keepdim=True)

    return train_sequences, valid_sequences, test_sequences
