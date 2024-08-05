import torch
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from criterion import Loss
from tqdm import tqdm
from matplotlib.colors import ListedColormap

from models.network import ANetwork
from datasets import ImageDataset, CustomSampler
from constants import TASKS, METRICS
from train import print_metrics
from utils.logger import get_logger, set_level
from utils.state_dict_utils import checkpoint_load
from utils.profiler import Profile
from utils.test_parser import parse_args

profiler = None

CLASS_NAMES = {
    'upper_color': ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'],
    'lower_color': ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'],
    'gender': ['Male', 'Female'],
    'bag': ['No', 'Yes'],
    'hat': ['No', 'Yes']
}

taskToName = {
    'upper_color': 'Upper Color',
    'lower_color': 'Lower Color',
    'gender' : 'Gender',
    'bag' : 'Bag',
    'hat' : 'Hat'
}


def save_confusion_matrix(cm, task, filename):

    # Create a custom colormap
    colors = ['#e17b75', '#95eb8a']  # Off-diagonal and diagonal colors
    cmap = ListedColormap([colors[0], colors[1]])

    # Create masks for diagonal and off-diagonal elements
    mask = np.zeros_like(cm, dtype=bool)
    np.fill_diagonal(mask, True)

    df_cm = pd.DataFrame(cm, index=CLASS_NAMES[task], columns=CLASS_NAMES[task])

    plt.figure(figsize=(8, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap=ListedColormap(colors[0]), cbar=False, mask=mask, annot_kws={"size": 12, "color": 'black'})
    # Apply the opposite mask for the diagonal elements
    sns.heatmap(df_cm, annot=True, fmt='d', cmap=ListedColormap(colors[1]), cbar=False, mask=1-mask, annot_kws={"size": 12, "color": 'black'})

    plt.title(f'{taskToName[task]}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    plt.close()

def evaluate(model, loader, criterion, device):
    times = []

    model.eval()
    with torch.no_grad():
        val_loss = []
        val_metrics_lists = {task:{} for task in TASKS}
        all_true_labels = {task:[] for task in TASKS}
        all_pred_labels = {task:[] for task in TASKS}
        
        for task in TASKS:
            for metric in METRICS:
                val_metrics_lists[task][metric] = []
        
        for X, y in tqdm(loader, desc='Test'):
            X = X.to(device)
            y = y.to(device).float()

            if profiler:
                # Profile the operation
                with profiler:
                    o = model(X)

                times.append(profiler.dt / len(X))
            else:
                o = model(X)

            val_loss.append(criterion.evaluate(o, y, y > -1, method="sum")[0])
            accs = criterion.compute_metrics(o, y)
            for task in TASKS:
                for metric in METRICS:
                    val_metrics_lists[task][metric].append(accs[task][metric])
                # Record true and predicted labels
                all_true_labels[task].extend(y[:, TASKS.index(task)].cpu().numpy().astype(int))
                all_pred_labels[task].extend(o[TASKS.index(task)].argmax(dim=1).cpu().numpy().astype(int))
                        
    if profiler:
        logger.info("Average time per image: {:.6f}".format(sum(times) / len(times)))

    val_loss = torch.tensor(val_loss).mean().detach().item()
    
    val_metrics = {}
    for task in TASKS: 
        val_metrics[task] = {}
        for metric in METRICS:
            val_metrics[task][metric] = torch.tensor(val_metrics_lists[task][metric]).mean().detach().item()
        # Compute and plot confusion matrix
        for i in range(len(all_true_labels[task])):
            all_true_labels[task][i] = CLASS_NAMES[task][all_true_labels[task][i]]
            all_pred_labels[task][i] = CLASS_NAMES[task][all_pred_labels[task][i]]

        cm = confusion_matrix(all_true_labels[task], all_pred_labels[task], labels = CLASS_NAMES[task])
        save_confusion_matrix(cm, task,  f'confusion_matrix_{task}.png')

    return val_loss, val_metrics

def main():
    args = parse_args()

    test_img_root = args.data + "/test/test_set"
    test_label = args.data + "/test/test_set.txt"

    global logger
    logger = get_logger()
    set_level(args.severity)

    logger.info("Using main device: " + args.device)

    model = ANetwork(train_bb=False, backbone_name=args.backbone, backbone_size=args.backbone_size, attention = args.attention)
    model.eval()
    model.to(args.device)
    logger.info("Number of parameters in the model: {}".format(model.num_parameters()))

    preprocessing = model.get_preprocessing()

    test_set = ImageDataset(image_folder=test_img_root,
                                label_csv=test_label,
                                transform=preprocessing)
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs, shuffle=True, num_workers=args.nw)

    criterion = Loss(device = args.device, heads_weights = [1,1,1,1,1])
    criterion.to(args.device)

    if args.checkpoint is not None:
        logger.info("Loading checkpoint {}...".format(args.checkpoint))
        checkpoint_load(model, args.checkpoint)
    else: 
        logger.error("No checkpoint provided, continuing with random weights")

    if args.profile:
        global profiler
        profiler = Profile()

    logger.info("Test Set\t{} samples".format(len(test_set)))

    loss, metrics = evaluate(model, test_loader, criterion, args.device)

    print_metrics(metrics)

if __name__ == '__main__':
    main()
