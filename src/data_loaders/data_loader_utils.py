import numpy as np
import torch


def _get_labels(dataset):
    """
    Extract the labels from the input dataset.

    :param dataset: PyTorch dataset
    :return: labels: Labels from the dataset
    """
    if hasattr(dataset, 'type') and dataset.type() == 'torch.LongTensor':
        labels = dataset
    elif 'tensors' in dataset.__dict__.keys():
        labels = dataset.tensors[1]
    elif 'train_labels' in dataset.__dict__.keys():
        labels = dataset.train_labels
    elif 'labels' in dataset.__dict__.keys():
        labels = dataset.labels
    elif 'dataset' in dataset.__dict__.keys():
        if 'dataset' not in dataset.dataset.__dict__.keys():
            if 'train_labels' in dataset.dataset.__dict__.keys():
                labels = [dataset.dataset.train_labels[idx] for idx in dataset.indices]
            elif 'targets' in dataset.dataset.__dict__.keys():
                labels = [dataset.dataset.targets[idx] for idx in dataset.indices]
            else:
                labels = [dataset.dataset.labels[idx] for idx in dataset.indices]

        else:
            if 'train_labels' in dataset.dataset.dataset.__dict__.keys():
                labels = [dataset.dataset.dataset.train_labels[idx] for idx in dataset.indices]
            elif 'targets' in dataset.dataset.dataset.__dict__.keys():
                labels = [dataset.dataset.dataset.targets[idx] for idx in dataset.indices]
            else:
                labels = [dataset.dataset.dataset.labels[idx] for idx in dataset.indices]

    elif 'targets' in dataset.__dict__.keys():
        labels = dataset.targets
    else:
        raise NotImplementedError

    return labels


def _replace_labels(dataset, labels):
    """
    Replace the labels in the given dataset with the input labels.

    :param dataset: Dataset in which to replace the labels
    :param labels: New labels for the dataset
    :return: dataset: dataset with the labels replaced
    """
    if 'tensors' in dataset.__dict__.keys():
        dataset.tensors[1] = labels
    elif 'train_labels' in dataset.__dict__.keys():
        dataset.train_labels = labels
    elif 'labels' in dataset.__dict__.keys():
        dataset.labels = labels
    elif 'dataset' in dataset.__dict__.keys():
        if hasattr(dataset.dataset, 'train_labels'):
            for i, idx in enumerate(dataset.indices):
                dataset.dataset.train_labels[idx] = labels[i]
        elif hasattr(dataset.dataset, 'labels'):
            for i, idx in enumerate(dataset.indices):
                dataset.dataset.labels[idx] = labels[i]
    else:
        raise NotImplementedError

    return dataset


def remove_labels(dataset=None, frac_labeled=None, num_labeled=None):
    """
    Remove labels from the data in the input dataset. Keep only frac_labeled or num_labeled sequences. This function
    assumes that all of the data is loaded into memory at once.

    :param dataset: The dataset from which some labels should be removed
    :param frac_labeled: The fraction of data that should be labeled
    :param num_labeled: The number of observations that should be labeled
    :return: Tuple containing:

            * dataset: Dataset with some of the labels removed
            * labeled_dataset: Dataset containing only the labeled sequences from dataset
            * unlabeled_dataset: Dataset containing only the unlabeled sequences from dataset
    """
    if frac_labeled is None and num_labeled is None:
        raise ValueError('Either the fraction of labeled data or the number of labeled data points must be specified')
    elif frac_labeled is not None and num_labeled is not None:
        raise ValueError('Only one of the fraction of labeled data or the number of labeled data points can be '
                         'specified')
    labels = _get_labels(dataset)

    if frac_labeled is not None:
        num_labeled = int(np.ceil(len(labels)*frac_labeled))
        max_num_labeled = int(len(labels)*frac_labeled)
        print('Number of labeled sequences being used:', max_num_labeled)
    else:
        max_num_labeled = num_labeled
        print('Number of labeled sequences being used:', max_num_labeled)

    labeled_idxs = np.random.choice(range(len(labels)), num_labeled, replace=False)[:max_num_labeled]
    labels_selected = [labels[idx] for idx in labeled_idxs]
    labels = [torch.ones_like(labels[i])*-1 for i in range(len(labels))]
    for i, idx in enumerate(labeled_idxs):
        labels[idx] = labels_selected[i]

    if 'dataset' not in dataset.__dict__.keys():
        unlabeled_idxs = [i for i in range(len(dataset)) if i not in labeled_idxs]
        labeled_dataset = torch.utils.data.dataset.Subset(dataset, labeled_idxs)
        unlabeled_dataset = torch.utils.data.dataset.Subset(dataset, unlabeled_idxs)
    else:
        dataset_indices = torch.LongTensor(dataset.indices)
        unlabeled_idxs = list(set(range(len(dataset_indices))).difference(set(labeled_idxs)))
        labeled_dataset = torch.utils.data.dataset.Subset(dataset.dataset, dataset_indices[labeled_idxs])
        unlabeled_dataset = torch.utils.data.dataset.Subset(dataset.dataset, dataset_indices[unlabeled_idxs])
    dataset = _replace_labels(dataset, labels)

    return dataset, labeled_dataset, unlabeled_dataset
