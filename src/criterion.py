import torch.nn as nn
import torch as t
import numpy as np
import json 
import warnings
import sys
sys.path.append('..')

from constants import *
from sklearn import metrics
from pathlib import Path

from utils.logger import get_logger
logger = get_logger()

warnings.filterwarnings("ignore", category=UserWarning)

class Loss():
    """
    Class to manage the loss functions.
    """

    def __init__(self, device, loss_name = "cross", **kwargs):

        self.device = device

        self.activation = nn.LogSoftmax(dim=-1) # TODO: Check if this is correct
        
        logger.debug("Creating loss: {}".format(loss_name))
        self.loss, self.activated_by_loss = loss_factory(loss_name, device, **kwargs)

        
    def __activate(self, input) -> [t.Tensor]:
        """
        Apply the activation function specified to the input.

        Parameters
        ----------
        input : tensor
            Tensor to apply the activation function to.

        Returns
        -------
        list
            Tensor with the activation function applied.
        """
        
        return self.activation(input)

    def evaluate(self, input: [t.Tensor], target: t.Tensor, mask) -> [t.Tensor]:
        """
        Evaluate the loss for each output using specified loss.
        The input MUST not be activated since the loss make its own activation.

        Parameters
        ----------
        input : tensor
            Tensor with shape BxC containing the logits output of the network.

        target : t.Tensor
            Tensor with shape B containing the target values.

        mask : t.Tensor
            Tensor with shape B containing the mask for the target values.

        Returns
        -------
        float
            The loss value.
        """
        device = target.device.type

        target = t.transpose(target, 0, 1) # TODO: Check if this is correct

        if len(mask.shape) != 1:
            mask = mask.squeeze(0)

        if not self.activated_by_loss:
            input = self.__activate(input)

        input = input.to(device)

        # Otherwise the masking will not work
        if len(input.shape) == 1:
            input = input.unsqueeze(0)

        # Find the labeled samples and compute the loss only on them
        l = self.loss(input[mask], input[mask])

        return l

    def compute_metrics(self, output: list, target: t.Tensor) -> [str, dict[str, float]]:
        """
        Compute some metrics for each output.
        The method expects the input not to be activated.

        Parameters
        ----------
        output : tensor
            Tensor containing the logits output of the network. BxC shape.

        target : t.Tensor
            Tensor containing the target values.

        Returns
        -------
        list
            Dict of metrics for each output task.
        """
        activations = self.__activate(output)

        metrics = self.__compute_metrics(activations, target)
        return metrics

    def __compute_metrics(self, output: t.Tensor, target: t.Tensor) -> dict:
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        binary = len(output[0]) <= 2

        # Determine class best predictions
        output = np.argmax(output, axis=1)

        metric = {
            'accuracy': metrics.accuracy_score(target, output),
            'balanced accuracy': metrics.balanced_accuracy_score(target, output),
            'precision': metrics.precision_score(target, output, average = 'binary' if binary else 'weighted'),
            'recall': metrics.recall_score(target, output, average = 'binary' if binary else 'weighted')
        }

        return metric

    def cuda(self):
        for loss in self.loss:
            loss.cuda()

    def cpu(self):
        for loss in self.loss:
            loss.cpu()

    def to(self, device):
        if device == 'cuda':
            self.cuda()
        if device == 'cpu':
            self.cpu()


def __create_cross(use_weights: bool, device):
    if use_weights: 
        with open(DATASET_FREQUENCIES_PATH) as f:
            frequencies = json.load(f)
            num_classes = len(frequencies)

            frequencies_list = [frequencies[f'{class_}'] for class_ in range(num_classes)]
            inverse_frequencies_list = [1/f for f in frequencies_list]

            weights = t.tensor(inverse_frequencies_list, device = device)
            logger.info("CrossEntropy weights: {}".format(inverse_frequencies_list))

            loss = nn.CrossEntropyLoss(weight = weights)
    else:
        loss = nn.CrossEntropyLoss()

    return loss

def loss_factory(loss_name: str, device, **kwargs):
    """
    Factory method to create the losses list.

    Parameters
    ----------
    loss_name : str
        Name of the loss to create.

    Returns
    -------
    list
        The losses list.

    bool
        Whether the loss automatically applies the activation function or not.
    """
    if loss_name == 'cross':
        use_weights = kwargs.get('use_weights', False)
        losses = __create_cross(use_weights, device)
        return losses, False
        # return [nn.CrossEntropyLoss() for _ in range(len(TASKS))], False
    elif loss_name == 'cross_custom':
        use_weights = kwargs.get('use_weights', True)
        losses = __create_cross(use_weights, device)
        return losses, False
    else:
        raise ValueError("Invalid loss name")


if __name__ == '__main__':
    Loss(loss_name='cross', device='mps')

