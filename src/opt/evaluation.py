import collections
import numpy as np
import torch

from chapydette import cp_estimation

from src import default_params as defaults
from src.opt import opt_utils, train_classifier


def evaluate_features(data, model, params):
    """
    Evaluate the current performance of the model using the given data.

    :param data: Data object containing the training, validation, and test set dataloaders
    :param model: Model object containing the architecture used in training
    :param params: Parameters object
    :return: Dictionary of results with the accuracy, loss, f1 scores, and other dissimilarity measures on each dataset
    """
    all_features = opt_utils.compute_all_features(data.train_labeled_loader, data.train_unlabeled_loader,
                                                  data.valid_loader, data.test_loader, model)
    if len(all_features['train_labeled']['x']) > 0 and torch.max(all_features['train_labeled']['y'][0]) > 0:
        w = train_classifier.train((all_features['train_labeled']['x'], all_features['train_labeled']['y']),
                                   (all_features['valid']['x'], all_features['valid']['y']), (None, None), None,
                                   params.num_classes, 100, loss_name='mnl', input_features=True)[5]
    else:
        w = None

    results = collections.Counter()
    dataset_names = ['train_labeled', 'train_unlabeled', 'valid', 'test']
    for dataset_name in dataset_names:
        X = all_features[dataset_name]['x']
        if 'y_true' in all_features[dataset_name]:
            y_true = all_features[dataset_name]['y_true']
            cps_true = all_features[dataset_name]['cps_true']
        else:
            y_true = all_features[dataset_name]['y']
            cps_true = all_features[dataset_name]['cps']
        if len(X) > 0:
            for i in range(len(X)):
                est_cps, obj = cp_estimation.mkcpe(X[i].numpy(), n_cp=len(cps_true[i]), kernel_type='linear',
                                                   min_dist=params.min_dist, return_obj=True)

                est_cps = est_cps.flatten()
                true_cps_i = cps_true[i].numpy()
                X[i] = X[i].to(defaults.device)

                results[dataset_name + '_loss'] += obj/len(X[i])
                results[dataset_name + '_penalized_loss'] += compute_penalized_loss(X[i], obj/len(X[i]),
                                                                                        model, params).item()
                results[dataset_name + '_hausdorff1'] += compute_hausdorff1(est_cps, true_cps_i)
                results[dataset_name + '_frobenius'] += compute_frobenius(est_cps, true_cps_i, len(X[i]))
                if w is not None:
                    results[dataset_name + '_accuracy'] += compute_num_correct_labels(X[i],
                                                                        y_true[i].to(defaults.device),
                                                                        torch.from_numpy(est_cps).to(defaults.device),
                                                                        w)
                else:
                    results[dataset_name + '_accuracy'] = np.nan
            for key in results.keys():
                if dataset_name in key:
                    if 'accuracy' not in key:
                        results[key] /= len(X)*1.0
                    else:
                        results[key] /= sum([len(y) for y in y_true])

        else:
            results[dataset_name + '_loss'] = np.inf
            results[dataset_name + '_penalized_loss'] = np.inf
            results[dataset_name + '_hausdorff1'] = np.inf
            results[dataset_name + '_frobenius'] = np.inf
            results[dataset_name + '_accuracy'] = 0

    return results


def compute_d1_infty(cps1, cps2):
    """
    Compute max_{1<=i<=cps1[-1]} {min_{1<=j<=cps2[-1]} |cps1[i]-cps2[j]|}.

    :param cps1: First array of change points
    :param cps2: Second array of change points
    :return: Maximum distance from a change point in cps1 to the nearest change point in cps2
    """
    pairwise_dists = np.abs(np.subtract.outer(cps1, cps2))
    min_dists = np.min(pairwise_dists, axis=1)
    return np.max(min_dists)


def compute_hausdorff1(cps1, cps2):
    """
    Compute the Hausdorff distance between sets of change points cps1 and cps2 wrt the distance d(x,y)=|x-y|, i.e.,
    max{d^(1)_\infty(cps1, cps2), d^(1)_\infty(cps1, cps2)}.

    :param cps1: First array of change points
    :param cps2: Second array of change points
    :return: Hausdorff distance between sets of change points cps1 and cps2 wrt the distance d(x,y)=|x-y|
    """
    return max(compute_d1_infty(cps1, cps2), compute_d1_infty(cps2, cps1))


def compute_projection_matrix(cps, sequence_length):
    """
    Compute the projection matrix P given by
    P_{i,i'} = 1/(cp[j+1]-cp[j]) if observations i and i' are both in segment j and 0 else
    (after having appended -1 and (sequence_length-1) to cps).

    :param cps: Change points to use to compute the projection matrix
    :param sequence_length: Length of the sequence being considered
    :return: projection_matrix: The projection matrix defined above
    """
    segment_lengths = np.diff(np.concatenate(([-1], cps, [sequence_length-1])))
    projection_matrix = np.zeros((sequence_length, sequence_length))
    prev_idx = 0
    for segment_length in segment_lengths:
        projection_matrix[prev_idx:prev_idx+segment_length, prev_idx:prev_idx+segment_length] = 1/segment_length
        prev_idx += segment_length

    return projection_matrix


def compute_frobenius(cps1, cps2, sequence_length):
    """
    Compute the Frobenius norm distance between the projection matrices corresponding to cps1 and cps2

    :param cps1: First array of change points
    :param cps2: Second array of change points
    :param sequence_length: Length of the sequence being considered
    :return: The Frobenius norm distance between the projection matrices corresponding to cps1 and cps2
    """
    segment_matrix1 = compute_projection_matrix(cps1, sequence_length)
    segment_matrix2 = compute_projection_matrix(cps2, sequence_length)

    return np.linalg.norm(segment_matrix1-segment_matrix2)


def compute_num_correct_labels(X, true_y, cps, w):
    """
    Given the classifier parameterized by w, predict the label of each segment using the mode of the predictions for the
    individual observations. Return the number of correctly labeled observations.

    :param X: Tensor of features for one sequence
    :param true_y: Tensor of true labels for the observations in X
    :param cps: Estimated change points
    :param w: Parameters of the classifier
    :return: total_correct: Number of correctly labeled observations
    """
    yhat = torch.max(torch.mm(X, w[1:, :]) + w[0, :], 1)[1]
    cps = torch.cat((torch.IntTensor([-1]).to(defaults.device), cps, torch.IntTensor([len(X)-1]).to(defaults.device)))
    total_correct = 0
    for i in range(1, len(cps)):
        est_y = torch.mode(yhat[cps[i-1]+1:cps[i]+1]).values
        est_y = est_y.item() if len(est_y.shape) == 0 else est_y[0]
        segment_length = cps[i]-cps[i-1]
        total_correct += torch.sum(torch.IntTensor([est_y]*segment_length).to(defaults.device) ==
                                   true_y[cps[i-1]+1:cps[i]+1]).item()

    return total_correct


def print_results(iteration, results, header=False):
    """
    Print the results at the current iteration.

    :param iteration: Current iteration number
    :param results: Dictionary with various measures of performance on the training, validation, and test sets
    :param header: Whether to print the column headers
    """
    if header:
        print('Iteration \t Test Frobenius \t Test loss \t Train-labeled Frobenius \t Train-labeled loss '
              '\t Train-unlabeled Frobenius \t Train-unlabeled loss')
    print(iteration, '\t\t',
          '{:06.4f}'.format(results['test_frobenius']), '\t',
          '{:06.4f}'.format(results['test_penalized_loss']), '\t'
          '{:06.4f}'.format(results['train_labeled_frobenius']), '\t',
          '{:06.4f}'.format(results['train_labeled_penalized_loss']), '\t',
          '{:06.4f}'.format(results['train_unlabeled_frobenius']), '\t',
          '{:06.4f}'.format(results['train_unlabeled_penalized_loss']),
          )


def compute_penalized_loss(X, obj, model, params):
    """
    Compute the loss (including the penalty terms)

    :param X: Features
    :param obj: Objective value output by chapydette
    :param model: Model object containing the architecture used in training
    :param params: Parameters object
    :return: penalized loss: The loss (including the penalty terms)
    """
    neg_trace_cov = opt_utils.segment_obj(X)
    if params.ckn:
        penalized_loss = obj - params.lambda_cov*torch.sqrt(-1*neg_trace_cov/len(X) + params.epsilon)
    else:
        sq_norm_params = 0
        for key in model.model.state_dict():
            sq_norm_params = sq_norm_params + torch.sum(model.model.state_dict()[key]**2)
        penalized_loss = obj - params.lambda_cov*torch.sqrt(-1*neg_trace_cov/len(X) + params.epsilon) + \
                         params.lambda_params*sq_norm_params

    return penalized_loss
