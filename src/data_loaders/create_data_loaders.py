import random
import torch
import torch.utils.data

from src import default_params as defaults
from src.data_loaders.data_loader_utils import remove_labels


def generate_dataloaders(train_dataset, test_dataset, valid_dataset=None, separate_valid_set=False, valid_size=0,
                         batch_size=1, batch_size_labeled=None, batch_size_unlabeled=None, frac_labeled=None,
                         num_labeled=None, num_workers=0):
    """
    Create data loaders given the corresponding PyTorch Dataset objects.

    :param train_dataset: Training dataset
    :param test_dataset: Test dataset
    :param valid_dataset: Validation dataset
    :param separate_valid_set: Whether to use a separate validation set or to use part of the training dataset
    :param valid_size: Validation set size
    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param frac_labeled: Fraction of the training set that should be labeled. Only one of frac_labeled and num_labeled
                         can be specified.
    :param num_labeled: Number of images in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param num_workers: Number of workers to use for the dataloaders
    :return: Tuple containing:

            * train_loader: Dataloader for the (labeled+unlabeled) training data
            * train_labeled_loader: Dataloader for the labeled training data.
            * train_unlabeled_loader: Dataloader for the unlabeled training data
            * valid_loader: Dataloader for the validation set
            * test_loader: Dataloader for the test set
    """
    if batch_size_labeled is None:
        batch_size_labeled = batch_size
    if batch_size_unlabeled is None:
        batch_size_unlabeled = batch_size

    if separate_valid_set:
        if frac_labeled is not None or num_labeled is not None:
            train_dataset, train_labeled_dataset, train_unlabeled_dataset = remove_labels(dataset=train_dataset,
                                                                              frac_labeled=frac_labeled,
                                                                              num_labeled=num_labeled)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout, collate_fn=collate_fn)
        valid_dataset.labels = valid_dataset.true_labels.copy()
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout, collate_fn=collate_fn)

        extra_train_valid_data = {}
        if train_dataset.additional_data:
            for key in train_dataset.additional_data:
                extra_train_valid_data[key] = train_dataset.additional_data[key] + valid_dataset.additional_data[key]

        if frac_labeled is not None or num_labeled is not None:
            if num_labeled > 0:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size_labeled,
                                                               shuffle=True, num_workers=num_workers,
                                                               pin_memory=True, drop_last=False,
                                                               timeout=defaults.dataloader_timeout,
                                                               collate_fn=collate_fn)
            else:
                train_labeled_loader = None
            if len(train_unlabeled_dataset) > 0:
                train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                     batch_size=batch_size_unlabeled,
                                                                     shuffle=True, num_workers=num_workers,
                                                                     pin_memory=True, drop_last=False,
                                                                     timeout=defaults.dataloader_timeout,
                                                                     collate_fn=collate_fn)
            else:
                train_unlabeled_loader = None
        else:
            train_labeled_loader = train_unlabeled_loader = None

    elif valid_size > 0:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        train_novalid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[valid_size:])
        valid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[:valid_size])

        if frac_labeled is not None or num_labeled is not None:
            train_novalid_dataset, train_labeled_dataset, train_unlabeled_dataset = remove_labels(
                                                                                      dataset=train_novalid_dataset,
                                                                                      frac_labeled=frac_labeled,
                                                                                      num_labeled=num_labeled)

        train_loader = torch.utils.data.DataLoader(train_novalid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout, collate_fn=collate_fn)

        if frac_labeled is not None or num_labeled is not None:
            if num_labeled > 0:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset,
                                                                   batch_size=batch_size_labeled,
                                                                   shuffle=True,
                                                                   num_workers=num_workers,
                                                                   pin_memory=True,
                                                                   drop_last=False,
                                                                   timeout=defaults.dataloader_timeout,
                                                                   collate_fn=collate_fn)
            else:
                train_labeled_loader = None
            train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                 batch_size=batch_size_unlabeled,
                                                                 shuffle=True,
                                                                 num_workers=num_workers,
                                                                 pin_memory=True,
                                                                 drop_last=False,
                                                                 timeout=defaults.dataloader_timeout,
                                                                 collate_fn=collate_fn)
        else:
            train_labeled_loader = train_unlabeled_loader = None
    else:
        if frac_labeled is not None or num_labeled is not None:
            train_dataset, train_labeled_dataset, train_unlabeled_dataset = remove_labels(dataset=train_dataset,
                                                                                      frac_labeled=frac_labeled,
                                                                                      num_labeled=num_labeled)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout, collate_fn=collate_fn)
        if frac_labeled is not None or num_labeled is not None:
            if num_labeled > 0:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size_labeled,
                                                               shuffle=True, num_workers=num_workers,
                                                               pin_memory=True, drop_last=False,
                                                               timeout=defaults.dataloader_timeout,
                                                               collate_fn=collate_fn)
            else:
                train_labeled_loader = None
            train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                 batch_size=batch_size_unlabeled,
                                                                 shuffle=True,
                                                                 num_workers=num_workers,
                                                                 pin_memory=True,
                                                                 drop_last=False,
                                                                 timeout=defaults.dataloader_timeout,
                                                                 collate_fn=collate_fn)
        else:
            train_labeled_loader = train_unlabeled_loader = None
        valid_loader = None

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, drop_last=False, pin_memory=True,
                                              timeout=defaults.dataloader_timeout, collate_fn=collate_fn)

    return train_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader


class PreloadedDataset(torch.utils.data.Dataset):
    """
    Dataset class for data that has already been loaded into memory that allows for transformations.

    :param features: Features to be in the dataset
    :param labels: Labels for the above features
    :param change_points: Change points for each sequence
    :param transform: Transformation that should be applied to the features in the dataset
    :param kwargs: Additional data to be returned with each sequence
    """
    def __init__(self, features, labels, change_points, transform=None, **kwargs):
        self.features = features
        self.labels = labels
        self.change_points = change_points
        self.transform = transform
        self.true_labels = labels.copy()
        self.true_change_points = change_points.copy()
        self.additional_data = kwargs

    def __getitem__(self, index):
        features = self.features[index]
        if self.transform is not None:
            features = self.transform(features)
        label = self.labels[index]
        true_label = self.true_labels[index]
        if label[0] != -1:
            change_points = self.change_points[index]
        else:
            change_points = []
        true_change_points = self.true_change_points[index]
        if self.additional_data is not None:
            additional_data = {}
            for key in self.additional_data:
                additional_data[key] = self.additional_data[key][index]
        else:
            additional_data = None

        return features, label, change_points, true_label, true_change_points, additional_data

    def __len__(self):
        return len(self.features)


def collate_fn(data):
    """
    Custom function that deals with a mini-batch of tensors as a list of tuples.

    :param data: List of tuples (x, y, change_points, true_y, true_change_points, additional_data)
    :return: Tuple containing:

            * x: Inputs, as either a tuple of tensors or a single tensor if the mini-batch size is 1
            * y: Labels, as either a tuple of tensors or a single tensor if the mini-batch size is 1
            * change_points: Change points, as either a tuple of tensors or a single tensor if the mini-batch size is 1
            * true_y: True labels (no unknown labels), as either a tuple of tensors or a single tensor if the
                      mini-batch size is 1
            * true_change_points: True change points (no unknown change points), as either a tuple of tensors or a
                                  single tensor if the mini-batch size is 1
            * additional_data: Any additional data associated with each sequence
    """
    return zip(*data)
