import os
import sys
import torch

from argparse import ArgumentParser


def parse_args():
    """
    Parse the arguments for the training script.
    The dataset organization requested is the following:
        args.data:
            |---> test
                    |->test_set.txt     (label csv file)
                    L->test_set         (folder containing samples)

    Returns
    -------
    args : argparse.Namespace
        The parsed arguments.
    """
    parser = ArgumentParser()

    # data
    parser.add_argument('--data', dest='data', help="Path to the dataset folder", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint", help="path to checkpoint to restore",
                        default=None, type=str)
    parser.add_argument('--device', dest='device', help="device to use", default='cpu', type=str)
    parser.add_argument('--bs', dest='bs', help="batch size", default=32, type=int)
    parser.add_argument('--nw', dest='nw', help="number of workers", default=4, type=int)
    parser.add_argument('--severity', dest='severity', help="logging severity",
                        default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--backbone', dest='backbone', help="Backbone to use",
                        default='convnext', type=str, choices=['swin', 'convnext', 'resnext', 'resnet'])
    parser.add_argument('--backbone_size', dest='backbone_size', help="Size of the backbone. Depends on the chosen backbone",
                        default='base', type=str)
    parser.add_argument('--attention', dest='attention', help="Attention module to use",
                        choices=['ham', 'cbam', 'none'], default='cbam', type=str)
    parser.add_argument('--profile', dest='profile', 
                        action="store_true", help="Profile the training by plotting mean time and memory consumed during validation",)

    args = parser.parse_args()

    return args

