import os
import sys
import torch

from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_hp import WORKERS, BATCH_SIZE, LEARNING_RATE, EPOCHS, OPTIMIZER

strToOptim = {
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD
}

def parse_args():
    """
    Parse the arguments for the training script.
    The dataset organization requested is the following:
        args.data:
            |---> train
            |       |->training_set.txt     (label csv file)
            |       L->training_set         (folder containing samples)
            |
            L---> val
                    |->validation_set.txt   (label csv file)
                    L->validation_set       (folder containing samples)

    Returns
    -------
    args : argparse.Namespace
        The parsed arguments.
    """
    parser = ArgumentParser()

    # checkpoint
    # The checkpoint name must comprise the string '_epoch_N' in order to start from epoch N
    parser.add_argument('--data', dest='data',
                        help="Path to the dataset folder", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint", help="path to checkpoint to restore",
                        default=None, type=str)

    # training hyper-parameters
    parser.add_argument('--nw', dest='nw', help="number of workers for dataloader",
                        default=WORKERS, type=int)
    parser.add_argument('--bs', dest='bs', help="batch size",
                        default=BATCH_SIZE, type=int)
    parser.add_argument('--lr', dest='lr', help="learning rate",
                        default=LEARNING_RATE, type=float)
    parser.add_argument('--device', dest='device', help="device to use",
                        default='cpu', type=str)
    parser.add_argument('--sched', dest='sched', action="store_true", help="Use or not use the learning rate scheduler",
                        default=False)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs to train",
                        default=EPOCHS, type=int)
    parser.add_argument('--loss', dest='loss', help="loss to use", choices=[ 'cross', 'cross_custom'],
                        default='cross', type=str)
    parser.add_argument('--train_bb', dest='train_bb', action="store_true", help="Train the network with the backbone",
                        default=True)
    parser.add_argument('--backbone', dest='backbone', help="Backbone to use",
                        default='convnext', type=str, choices=['swin', 'convnext', 'resnext', 'resnet'])
    parser.add_argument('--backbone_size', dest='backbone_size', help="Size of the backbone. Depends on the chosen backbone",
                        default='base', type=str)
    parser.add_argument('--attention', dest='attention', help="Attention module to use",
                        choices=['ham', 'cbam', 'none'], default='ham', type=str)
    parser.add_argument('--severity', dest='severity', help='Logging severity level',
                        default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--profile', dest='profile', action="store_true", 
                        help="Profile the training by plotting mean time and memory consumed during validation",)
    parser.add_argument('--optim', dest='optim', help="Optimizer to use", 
                        type=str, choices=['adamw', 'sgd']) # TODO: enhance


    args = parser.parse_args()

    return args
