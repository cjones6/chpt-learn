import random
import torch

from src import default_params as defaults


def get_data(data, iteration, batch_size=None):
    """
    Extract the data, labels, and change points from the labeled and unlabeled training sets.

    :param data: Data object containing the training, validation, and test set dataloaders
    :param iteration: Present training iteration (used for setting a random seed)
    :param batch_size: Size of the mini-batch to return (if None, it will return the entire training dataset)
    :return: Tuple containing:

        * x: The observations in the mini-batch
        * cps: The known change points (for the unlabeled data it contains empty lists)
        * cps_truth: The true change points in the mini-batch
    """
    if data.train_unlabeled_loader:
        x_unlabeled, _, cps_unlabeled, _, cps_unlabeled_truth, _ = zip(*[batch for batch in data.train_unlabeled_loader])
        x_unlabeled = sum([list(x_unlabeled[i]) for i in range(len(x_unlabeled))], [])
        cps_unlabeled = sum([list(cps_unlabeled[i]) for i in range(len(cps_unlabeled))], [])
        cps_unlabeled_truth = sum([list(cps_unlabeled_truth[i]) for i in range(len(cps_unlabeled_truth))], [])
    else:
        x_unlabeled, cps_unlabeled, cps_unlabeled_truth = [], [], []

    if data.train_labeled_loader:
        x_labeled, _, cps_labeled, _, _, _ = zip(*[batch for batch in data.train_labeled_loader])
        x_labeled = sum([list(x_labeled[i]) for i in range(len(x_labeled))], [])
        cps_labeled = sum([list(cps_labeled[i]) for i in range(len(cps_labeled))], [])
    else:
        x_labeled, cps_labeled = [], []

    x = x_labeled + x_unlabeled
    cps = cps_labeled + cps_unlabeled
    cps_truth = cps_labeled + cps_unlabeled_truth

    if batch_size is not None:
        random.seed(iteration)
        random.shuffle(x)
        random.seed(iteration)
        random.shuffle(cps)
        random.seed(iteration)
        random.shuffle(cps_truth)
        x = x[:batch_size]
        cps = cps[:batch_size]
        cps_truth = cps_truth[:batch_size]

    x = tuple(x)
    cps = tuple(cps)
    cps_truth = tuple(cps_truth)

    return x, cps, cps_truth


def compute_features(x, model):
    """
    Compute features for a mini-batch of inputs.

    :param x: Inputs to the network as a single tensor or tuple of tensors
    :param model: Model (object from a subclass of the nn.Module class)
    :return: features_list: List of computed features for each input sequence
    """
    if isinstance(x, tuple):
        sequence_lengths = [len(x[j]) for j in range(len(x))]
        x = torch.cat(x).type(torch.get_default_dtype()).to(defaults.device)
    else:
        sequence_lengths = [len(x)]
        x = x.type(torch.get_default_dtype()).to(defaults.device)
    features = model(x)
    features = features.reshape(features.shape[0], -1)

    idx = 0
    features_list = []
    for j, sequence_length in enumerate(sequence_lengths):
        features_list.append(features[idx:idx + sequence_length].view(sequence_length, -1))
        idx += sequence_length

    return features_list


def compute_all_features(train_lab_loader, train_unlab_loader, valid_loader, test_loader, model):
    """
    Generate features for all inputs in the training, validation, and test sets using the given model.

    :param train_lab_loader: Dataloader for the labeled training data
    :param train_unlab_loader: Dataloader for the unlabeled training data
    :param valid_loader: Dataloader for the validation data
    :param test_loader: Dataloader for the test data
    :param model: Model (object from a subclass of the nn.Module class)
    :return: all_features: Dictionary with the features and labels for each of the input datasets
    """
    with torch.autograd.set_grad_enabled(False):
        all_features = {'train_unlabeled': {'x': [], 'y': [], 'y_true': [], 'cps': [], 'cps_true': []},
                        'train_labeled': {'x': [], 'y': [], 'cps': []},
                        'valid': {'x': [], 'y': [], 'cps': []},
                        'test': {'x': [], 'y': [], 'cps': []}}
        for dataset_name, data_loader in zip(sorted(all_features.keys()), [test_loader, train_lab_loader,
                                                                           train_unlab_loader, valid_loader]):
            if data_loader is not None:
                for i, (x, y, cps, ytrue, cpstrue, _) in enumerate(data_loader):
                    sequence_lengths = [len(x[j]) for j in range(len(x))]
                    # Concatenate the features from different sequences
                    cat_features = torch.cat(x).type(torch.get_default_dtype()).to(defaults.device)
                    if model is not None:
                        features = model(cat_features)
                    else:
                        features = cat_features
                    # Separate out the features from the different sequences and reshape the output so there is one
                    # feature vector per observation in a sequence
                    idx = 0
                    features_list = []
                    for j, sequence_length in enumerate(sequence_lengths):
                        features_list.append(features[idx:idx+sequence_length].view(sequence_length, -1).data.cpu())
                        idx += sequence_length
                    all_features[dataset_name]['x'].extend(features_list)
                    all_features[dataset_name]['y'].extend(y)
                    all_features[dataset_name]['cps'].extend(cps)
                    if 'y_true' in all_features[dataset_name]:
                        all_features[dataset_name]['y_true'].extend(ytrue)
                        all_features[dataset_name]['cps_true'].extend(cpstrue)

    return all_features


def compute_normalizations(model):
    """
    Compute the term k(W^TW)^{-1/2} for each layer.

    :param model: CKN model
    :return model: Model after the normalizations k(W^TW)^{-1/2} have been (re)computed and stored
    """
    for layer_num in range(len(model.layers)):
        model.layers[layer_num].store_normalization = True
        with torch.autograd.set_grad_enabled(False):
            model.layers[layer_num].normalization = model.layers[layer_num].compute_normalization()

    return model


def segment_obj(X):
    """
    Compute the (unnormalized) objective on an individual segment

    :param X: Segment on which to compute the objective
    :return: Negative of the squared norm of the difference between X and its mean
    """
    return -torch.norm(X - torch.mean(X, 0))**2


def compute_obj(X, cps, params):
    """
    Compute the objective on X with the given set of change points cps, without the penalty on the network weights. That
    penalty will be added during the optimization.

    :param X: Tensor or tuple of tensors on which to compute the objective
    :param cps: Tensor or tuple of tensors of change points corresponding to X
    :param params: Parameter object
    :return: total_obj: Objective averaged over all input sequences in X
    """
    if isinstance(X, tuple):
        total_obj = 0
        for i in range(len(X)):
            obj = segment_obj(X[i][0:cps[i][0] + 1])
            for j in range(1, len(cps[i])):
                obj = obj + segment_obj(X[i][cps[i][j-1] + 1:cps[i][j] + 1])
            obj = obj + segment_obj(X[i][cps[i][-1] + 1:])
            total_obj = total_obj + obj/len(X[i]) + params.lambda_cov*torch.sqrt(-1*segment_obj(X[i])/len(X[i]) +
                                                                                 params.epsilon)
        total_obj = total_obj/len(X)
    else:
        total_obj = segment_obj(X[0:cps[0]+1])
        for i in range(1, len(cps)):
            total_obj = total_obj + segment_obj(X[cps[i-1]+1:cps[i]+1])
        total_obj = total_obj + segment_obj(X[cps[-1]+1:])
        total_obj = total_obj/len(X) + params.lambda_cov*torch.sqrt(-1*segment_obj(X)/len(X) + params.epsilon)

    return total_obj


def one_hot_embedding(y, num_dims):
    """
    Generate a one-hot representation of the input vector y. From
    https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23

    :param y: Labels for which a one-hot representation should be created
    :param num_dims: Number of unique labels
    :return: One-hot representation of y
    """
    y_tensor = y.data.type(torch.LongTensor).view(-1, 1)
    num_dims = num_dims if num_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], num_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.shape[0], -1)

    return y_one_hot.type(torch.get_default_dtype()).to(defaults.device)
