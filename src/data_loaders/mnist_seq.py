import numpy as np
import torch
import torchvision

from . import create_data_loaders


def get_dataloaders(batch_size=1, batch_size_labeled=None, batch_size_unlabeled=None, data_path='../data/MNIST',
                    frac_labeled=None, max_cp=10, min_dist=5, num_labeled=None, num_train=None, num_workers=4,
                    seed=None, sequence_length=100, transform='std', valid_size=100):
    """
    Create dataloaders that load sequences of MNIST digits that will be used for change-point estimation.

    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param data_path: Directory where the data either currently exists or should be downloaded to
    :param frac_labeled: Fraction of sequences in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param max_cp: Maximum number of change points in a sequence
    :param min_dist: Minimum allowable distance between change points
    :param num_labeled: Number of sequences in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param num_train: Number of sequences in the training set. If None, it will be approximately 500.
    :param num_workers: Number of workers to use for the dataloaders
    :param seed: Seed to use when creating the training and test sequences
    :param sequence_length: Length of each sequence of digits
    :param transform: How the raw data should be transformed. Either 'std' (standardized) or None.
    :param valid_size: Number of sequences in the validation set
    :return: Tuple containing:

            * train_loader: Dataloader for the (labeled+unlabeled) training data
            * train_labeled_loader: Dataloader for the labeled training data.
            * train_unlabeled_loader: Dataloader for the unlabeled training data
            * valid_loader: Dataloader for the validation set
            * test_loader: Dataloader for the test set
    """
    def std(x):
        return (x.type(torch.get_default_dtype())/255.0-0.1307)/0.3081

    if transform == 'std':
        transform = std
    elif transform is not None:
        raise NotImplementedError

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    train_sequences, train_sequences_labels, train_sequences_cps = generate_sequences(train_dataset, sequence_length,
                                                                                      max_cp, min_dist, seed)
    test_sequences, test_sequences_labels, test_sequences_cps = generate_sequences(test_dataset, sequence_length,
                                                                                   max_cp, min_dist, seed+1)

    if num_train is not None:
        train_sequences = train_sequences[:num_train+valid_size]
        train_sequences_labels = train_sequences_labels[:num_train+valid_size]
        train_sequences_cps = train_sequences_cps[:num_train+valid_size]

    train_dataset = create_data_loaders.PreloadedDataset(train_sequences,
                                                         train_sequences_labels,
                                                         train_sequences_cps,
                                                         transform=transform)
    test_dataset = create_data_loaders.PreloadedDataset(test_sequences,
                                                        test_sequences_labels,
                                                        test_sequences_cps,
                                                        transform=transform)

    return create_data_loaders.generate_dataloaders(train_dataset,
                                                    test_dataset,
                                                    separate_valid_set=False,
                                                    valid_size=valid_size,
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
    possible_cps = range(min_dist-1, sequence_length-min_dist-1)
    while cp_idx < n_cp:
        possible_cp = np.random.choice(possible_cps)
        if np.min(np.abs(cps-possible_cp)) >= min_dist:
            cps[cp_idx] = possible_cp
            cp_idx += 1

    return sorted(cps)


def select_labels(cps, data_dict, sequence_length, min_dist=5):
    """
    Select which label will go in each segment.

    :param cps: Locations of the change points
    :param data_dict: Dictionary with the labels as keys and unused input data in the values
    :param sequence_length: Length of the sequence
    :param min_dist: Minimum allowable distance between change points
    :return: Tuple containing:

            * labels: Label for each segment
            * cps: Change points (which might've changed if edge cases were encountered)
    """
    # Store the number of each label still available
    num_left = [len(data_dict[label]) for label in sorted(data_dict.keys())]
    # Compute the length of each segment
    segment_lengths = compute_segment_lengths(cps, sequence_length)
    labels = []
    segment_idx = 0
    prev_label = -1

    while segment_idx < len(segment_lengths):
        # Propose a label for the next segment
        label = np.random.choice(max(data_dict.keys())+1)
        if label != prev_label:
            # If the label is not the same as the previous one and either there will be at least min_dist of this label
            # left after this or there will be none of this label left after this, then use it.
            if num_left[label] >= segment_lengths[segment_idx] + min_dist \
                    or num_left[label] == segment_lengths[segment_idx]:
                labels.append(label)
                num_left[label] -= segment_lengths[segment_idx]
                segment_idx += 1
                prev_label = label
            # If the label is not the same as the previous one but there will be between 0 and min_dist of this label
            # left then use it if this is not the last segment or if it is the last segment but the number left is less
            # than the segment length and this label is one of the two with the largest number of inputs remaining
            elif min_dist <= num_left[label] < segment_lengths[segment_idx] + min_dist:
                if segment_idx != len(segment_lengths)-1:
                    labels.append(label)
                    # Shift the change point and change the segment lengths based on the number left
                    cps[segment_idx] += num_left[label] - segment_lengths[segment_idx]
                    segment_lengths[segment_idx+1] += segment_lengths[segment_idx] - num_left[label]
                    segment_lengths[segment_idx] = num_left[label]
                    num_left[label] = 0
                    segment_idx += 1
                    prev_label = label
                elif np.max(num_left) < segment_lengths[segment_idx] or (np.argmax(num_left) == prev_label and \
                                                                sorted(num_left)[-2] < segment_lengths[segment_idx]):
                    labels.append(label)
                    # Add a new change point and segment length
                    if segment_idx >= 1:
                        cps.append(cps[segment_idx-1] + num_left[label])
                    elif segment_idx == 0:
                        cps.append(num_left[label] - 1)
                    segment_lengths = np.concatenate((segment_lengths, [segment_lengths[segment_idx]-num_left[label]]))
                    segment_lengths[segment_idx] = num_left[label]
                    num_left[label] = 0
                    segment_idx += 1
                    prev_label = label
            # If there are no inputs left, then we're done
            elif sum(num_left) == 0 and len(labels) == 1:
                cps = []
                segment_idx = len(segment_lengths)
            # Otherwise if the number left of the proposed label is between 0 and min_dist or there are no labels and
            # there aren't any possible labels, there is a problem
            elif (0 < num_left[label] < min_dist) or (sum(num_left) == 0 and len(labels) == 0):
                raise ValueError
        # If the label is the same as the previous one but is the last label left, then use it and delete any remaining
        # change points
        elif label == prev_label and num_left[label] == sum(num_left):
            del cps[segment_idx-1:]
            segment_idx = len(segment_lengths)
        # If the label is the same as the previous one and the label that has the second-most remaining inputs doesn't
        # have at least min_dist more than the segment length or isn't equal to the segment length and the number left
        # of this label is at least min_dist more than the segment length, then use it.
        elif label == prev_label and sorted(num_left)[-2] < segment_lengths[segment_idx] + min_dist and \
                sorted(num_left)[-2] != segment_lengths[segment_idx] and \
                num_left[label] >= segment_lengths[segment_idx] + min_dist:
            del cps[segment_idx - 1]
            num_left[label] -= segment_lengths[segment_idx]
            segment_lengths[segment_idx-1] += segment_lengths[segment_idx]
            segment_lengths = np.delete(segment_lengths, segment_idx)

    return labels, cps


def compute_segment_lengths(cps, sequence_length):
    """
    Given the length of a sequence and the locations of change points in that sequence, compute the length of each
    segment.

    :param cps: Change point locations (last element in each segment)
    :param sequence_length: Length of the sequence the change points are in
    :return: lengths: Lengths of the segments
    """
    cps = [-1] + cps + [sequence_length-1]
    lengths = np.diff(cps)

    return lengths


def generate_sequences(dataset, sequence_length, max_cp=10, min_dist=5, seed=0):
    """
    Generate sequences of inputs in which successive entries are usually from the same class.

    :param dataset: Dataset from which to construct sequences
    :param sequence_length: Length of desired sequences
    :param max_cp: Maximum number of desired change points per sequence
    :param min_dist: Minimum distance between change points
    :param seed: Random seed to use
    :return: Tuple containing:

            * sequences: Sequences of inputs of length sequence_length
            * sequences_labels: Sequences of labels corresponding to the above input sequences
            * sequences_cps: List of change points in the labels in the above sequences
    """
    np.random.seed(seed)
    nseq = len(dataset)//sequence_length

    # Store all input data by label in a dictionary where the keys are the labels
    data_dict = {}
    for i in range(torch.max(dataset.targets).item()+1):
        data_dict[i] = dataset.data[np.where(dataset.targets == i)[0]]

    sequences = []
    sequences_labels = []
    sequences_cps = []

    for i in range(nseq):
        # Choose the number of change points
        n_cp = np.random.choice(range(1, max_cp+1))
        # Select the locations of those change points
        cps = select_cp_locations(n_cp, sequence_length, min_dist=min_dist)
        # Select the labels that will go in each segment
        labels, cps = select_labels(cps, data_dict, sequence_length, min_dist=min_dist)
        segment_lengths = compute_segment_lengths(cps, sequence_length)
        # Create a sequence of inputs based on the previously selected labels and change points
        sequence = []
        sequence_labels = []
        for label, segment_length in zip(labels, segment_lengths):
            sequence.append(data_dict[label][0:segment_length].clone())
            data_dict[label] = data_dict[label][segment_length:]
            sequence_labels.extend([label]*segment_length)

        # Check the resultant sequences
        if len(cps) > 0:
            assert np.min(np.abs(np.diff(np.asarray(sequence_labels))[cps])) > 0
            assert np.max(np.abs(np.diff(np.asarray(sequence_labels))
                                 [list(set(range(len(sequence_labels)-1)).difference(cps))])) == 0
        else:
            assert np.max(np.abs(np.diff(np.asarray(sequence_labels)))) == 0
        if len(cps) > 1:
            assert np.min(np.diff(cps)) >= 1
        assert len(sequence_labels) == sequence_length
        assert len(torch.cat(sequence)) == sequence_length

        # Store the sequence if it has at least one change point (note that the number of change points can change in
        # the function select_labels if edge cases are encountered) and the minimum distance between change points is
        # at least 5
        if len(cps) > 0 and np.min(compute_segment_lengths(cps, sequence_length)) >= min_dist:
            sequences.append(torch.cat(sequence))
            sequences_labels.append(torch.IntTensor(sequence_labels))
            sequences_cps.append(torch.IntTensor(cps))

    return sequences, sequences_labels, sequences_cps
