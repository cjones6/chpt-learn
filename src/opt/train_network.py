import time
import torch
import torch.optim as optim

from chapydette import cp_estimation

from src import default_params as defaults
from . import evaluation, opt_utils


class TrainSupervised:
    """
    Class used to optimize the parameters of a network, parameters of a classifier, and any unknown labels.

    :param data: Data object containing the training, validation, and test set dataloaders
    :param model: Model object containing the architecture to be used in training
    :param params: Parameters object that contains the parameters related to training and evaluation
    :param results: Results object for storing the results
    """
    def __init__(self, data, model, params, results):
        self.data = data
        self.model = model
        self.params = params
        self.results = results

        self.iteration = 0
        self.step_size = params.lr
        self.params.only_unsup = self.data.train_labeled_loader is None

        if self.params.train_w_layers is None and self.params.ckn:
            self.params.train_w_layers = range(len(self.model.model.layers))
        if not self.params.ckn:
            self.optimizer = optim.SGD(self.model.model.parameters(), lr=self.params.lr,
                                       weight_decay=self.params.lambda_params)
        else:
            self.optimizer = None

    def _get_lr(self):
        """
        Get the learning rate to use at this iteration.
        """
        if self.params.lr_schedule is not None:
            return self.params.lr_schedule(self.iteration)
        else:
            return self.params.lr

    def _update_params(self):
        """
        Take one step to optimize the parameters at each layer.
        """
        self.step_size = self._get_lr()
        x, cps, cps_truth = opt_utils.get_data(self.data, self.iteration, batch_size=self.params.batch_size)
        self._take_step(x, cps, [len(cps_truth[i]) for i in range(len(cps_truth))])

    def _take_step(self, x, cps, ncp):
        """
        Take a step to update the parameters.
        """
        obj_value = self._alternating_optimization(x, cps, ncp)
        self.results.update(self.iteration, **{'obj': obj_value.cpu().detach().item()})
        if self.params.ckn:
            with torch.autograd.set_grad_enabled(False):
                for layer_num in self.params.train_w_layers:
                    grad = self.model.model.layers[layer_num].W.grad.data
                    W = self.model.model.layers[layer_num].W.data
                    if self.params.project:
                        W = W - self.step_size * grad / torch.norm(grad, 2, 1, keepdim=True)
                        W_proj = W / torch.norm(W, 2, 1, keepdim=True)
                        self.model.model.layers[layer_num].W.data = W_proj
                    else:
                        grad += 2*self.params.lambda_params*self.model.model.layers[layer_num].W
                        self.model.model.layers[layer_num].W.data = W - self.step_size * grad
        else:
            self.optimizer.step()

    def _alternating_optimization(self, x, cps, ncp):
        """
        Estimate the change points for the given features if they are unknown and then return the resultant objective.
        """
        self.model.zero_grad()
        obj_value = 0
        for i in range(len(x)):
            features = opt_utils.compute_features(x[i], self.model)[0]

            if len(cps[i]) == 0:
                with torch.autograd.no_grad():
                    est_cps = cp_estimation.mkcpe(features.cpu().numpy(), n_cp=ncp[i], kernel_type='linear',
                                              min_dist=self.params.min_dist, return_obj=False).ravel()
            else:
                est_cps = cps[i]

            obj_value = obj_value - opt_utils.compute_obj(features, est_cps, self.params)

        obj_value = obj_value/len(x)
        obj_value.backward()

        return obj_value

    def _evaluate(self):
        """
        Compute the change points on the full dataset and then evaluate the performance.
        """
        self.model.eval()
        if self.params.ckn:
            self.model.model = opt_utils.compute_normalizations(self.model.model)

        results = evaluation.evaluate_features(self.data, self.model, self.params)
        if self.iteration == 0:
            evaluation.print_results(self.iteration, results, header=True)
        else:
            evaluation.print_results(self.iteration, results, header=False)
        self.results.update(self.iteration, **results)

        if self.params.ckn:
            for layer_num in range(len(self.model.model.layers)):
                self.model.model.layers[layer_num].store_normalization = False
        self.model.train()

    def train(self):
        """
        Train the network and estimate the change points for each sequence.
        """
        iter_since_last_eval = iter_since_last_save = 0
        self._evaluate()

        while self.iteration < self.params.num_iters:
            t1 = time.time()
            self._update_params()
            t2 = time.time()
            self.iteration += 1
            self.results.update(self.iteration, epoch_time=t2-t1)
            iter_since_last_eval += 1
            iter_since_last_save += 1

            if iter_since_last_eval >= self.params.eval_test_every:
                self._evaluate()
                if iter_since_last_save >= self.params.save_every:
                    self.model.save(iteration=self.iteration, step_size=self.step_size, optimizer=self.optimizer)
                    self.results.save()
                    self.params.save()
                    iter_since_last_save = 0
                iter_since_last_eval = 0

        print('Done training. Saving final results.')
        self._evaluate()
        self.model.save(iteration=self.iteration, step_size=self.step_size, optimizer=self.optimizer)
        self.results.save()
        self.params.save()
