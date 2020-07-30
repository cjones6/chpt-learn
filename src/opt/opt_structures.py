import os
import pickle
import torch
import torch.nn as nn


class Data:
    """
    Class for storing the data loaders.

    :param train_labeled_loader: Dataloader for the labeled training set
    :param train_unlabeled_loader: Dataloader for the unlabeled training set
    :param valid_loader: Dataloader for the validation set
    :param test_loader: Dataloader for the test set
    """
    def __init__(self, train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader):
        self.train_labeled_loader = train_labeled_loader
        self.train_unlabeled_loader = train_unlabeled_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        if self.train_labeled_loader is not None:
            self.train_labeled_iter = iter(self.train_labeled_loader)
        else:
            self.train_labeled_iter = None
        if self.train_unlabeled_loader is not None:
            self.train_unlabeled_iter = iter(self.train_unlabeled_loader)
        else:
            self.train_unlabeled_iter = None


class Model(nn.Module):
    """
    Class that stores the model, evaluates the model on inputs, saves the model, and loads the model.

    :param model: Model (object from a subclass of the nn.Module class)
    :param save_path: Where to save the model. If None, the model is not saved.
    """
    def __init__(self, model=None, save_path=None):
        super(Model, self).__init__()
        self.model = model
        self.save_path = save_path

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def forward(self, x):
        """
        Generate features for the provided inputs with the model.

        :param x: Inputs for which features should be generated
        :return: Features for inputs x
        """
        if self.model is not None:
            return self.model(x)
        else:
            return x

    def save(self, **kwargs):
        """
        Save the model to a file. It assumes that the extension is '.pickle'.

        :param kwargs: Additional keyword arguments to save along with the model
        """
        if self.save_path is not None:
            save_dict = {'model': self.model}
            for key, value in kwargs.items():
                save_dict[key] = value
            torch.save(save_dict, self.save_path[:-7] + '_' + str(kwargs['iteration']) + self.save_path[-7:])

    def load(self, path):
        """
        Load the model from a file.

        :param path: Filepath of the model
        :return: model_dict: Dictionary with the model and anything else stored with it
        """
        model_dict = torch.load(path)
        self.model = model_dict['model']
        return model_dict


class Params:
    """
    Class to store a variety of hyperparameter values and input options.

    :param batch_size: Mini-batch size to use in training
    :param ckn: Whether the model is a CKN
    :param epsilon: Value of epsilon in the first penalty term
    :param eval_test_every: How often to evaluate the performance (after how many iterations)
    :param lambda_cov: Value of the l2 penalty on the square root of the trace of the covariance matrix
    :param lambda_params: Value of the l2 penalty on the parameters
    :param lr: Learning rate
    :param lr_schedule: Function used to decay the learning rate
    :param min_dist: Minimum allowable distance between change points
    :param num_classes: Number of classes in the dataset
    :param num_iters: Maximum number of iterations
    :param project: Whether to project the parameters onto a product of unit spheres
    :param save_every: How often to save the model, parameters, and results (after how many iterations)
    :param save_path: Location where the parameters should be saved. If None, they aren't saved
    :param train_w_layers: List of layers to train. If None, it trains the parameters of all of the layers.
    """
    def __init__(self, batch_size=None, ckn=True, epsilon=0, eval_test_every=1, lambda_cov=None, lambda_params=0,
                 lr=None, lr_schedule=None, min_dist=1, num_classes=10, num_iters=100, project=False, save_every=100,
                 save_path=None, train_w_layers=None):

        self.batch_size = batch_size
        self.ckn = ckn
        self.epsilon = epsilon
        self.eval_test_every = eval_test_every
        self.lambda_cov = lambda_cov
        self.lambda_params = lambda_params
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.min_dist = min_dist
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.project = project
        self.save_every = save_every
        self.save_path = save_path
        self.train_w_layers = train_w_layers

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def save(self):
        """
        Save the parameters to a file.
        """
        save_dir = os.path.join(*self.save_path.split(os.sep)[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def load(self, path):
        """
        Load the parameters from a file.

        :param path: Filepath of the parameters
        """
        params = pickle.load(open(path, 'rb'))
        for key, value in params.items():
            self.__dict__[key] = value


class Results:
    """
    Class to store the results from each iteration.

    :param save_path: Where to save the results. If None, they are not saved.
    """
    def __init__(self, save_path=None):
        self.save_path = save_path

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def update(self, iteration, **kwargs):
        """
        Update the dictionaries of results.

        :param iteration: Iteration number
        :param kwargs: Additional keyword arguments to save
        """
        for key, value in kwargs.items():
            if key not in self.__dict__:
                self.__dict__[key] = {}
            self.__dict__[key][iteration] = value

    def save(self):
        """
        Save the current results.
        """
        if self.save_path is not None:
            pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def load(self, path):
        """
        Load results from a file.

        :param path: Filepath of the results
        """
        params = pickle.load(open(path, 'rb'))
        for key, value in params.items():
            self.__dict__[key] = value
