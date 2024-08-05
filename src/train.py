import sys
sys.path.append("..")
import torch
import re
import warnings
import traceback
import logging

from time import localtime, strftime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ImageDataset
from models.network import ANetwork
from criterion import Loss
from train_hp import *
from constants import METRICS, BOT_TOKEN
from utils.state_dict_utils import checkpoint_save, checkpoint_load
from utils.train_plots import plot_results, print_metrics
from utils.train_parser import parse_args, strToOptim
from utils.telegram_bot import TelegramBot, update_telegram
from utils.logger import get_logger, set_level
from utils.profiler import Profile

global bot 
bot = TelegramBot(token = BOT_TOKEN)

global profiler
profiler = None

logger = None 
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

torch.autograd.set_detect_anomaly(False)

def mean(l):
    return sum(l) / len(l)

def one_epoch(model, criterion, optimizer, train_loader, val_loader, 
              device):

    model.train()
    batches_done = 0

    batch_losses = [] # necessary to give DWA the mean loss during epoch
       
    for X, y in tqdm(train_loader, desc='Train'):

        X = X.to(device).float()
        y = y.to(device).float()

        # Forward pass
        o = model(X)
        
        loss = criterion.evaluate(o, y, y > -1) # Get the loss value

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batches_done += 1

    # if batches_done != train_loader.batch_sampler.sampler.num_batches:
    #     warnings.warn(
    #         "Number of batches done during training is not sufficient to cover the whole training set. "
    #         "{} out of {} done.".format(batches_done, train_loader.batch_sampler.sampler.num_batches),
    #         RuntimeWarning
    #     )

    model.eval()

    with torch.no_grad():
        val_loss = []

        if profiler:
            times = []
            memory = []

        # contains different metrics for each batch
        val_metrics_lists = {}

        for metric in METRICS:
            val_metrics_lists[metric]=[]

        for X, y in tqdm(val_loader, desc='Validation'):
            X = X.to(device)
            y = y.to(device).float()


            if profiler:
                with profiler:
                    o = model(X)

                times.append(profiler.dt / len(X))
                memory.append(profiler.memory() / len(X))
            else:
                o = model(X)

            val_loss.append(criterion.evaluate(o, y, y > -1))

            # Compute the accuracy for each task, returns a list of 5 elements
            metrics = criterion.compute_metrics(o, y)

            # Append the accuracy for the current batch to the list of accuracies
            for metric in METRICS:
                val_metrics_lists[metric].append(metrics[metric])

    if profiler:
        logger.info("Average time per image: {:.4f}".format(mean(times)))

    val_loss = mean(val_loss)

    val_metrics = {}
    for metric in METRICS:
        val_metrics[metric] = mean(val_metrics_lists[metric])

    return val_loss, val_metrics

def train(model, start_epoch, epochs, lr, train_loader, val_loader, 
          criterion, device, experiment_name, scheduler=False):
    model.train()

    optimizer = OPTIMIZER(
        model.parameters(),
        lr = lr,
        # eps = 1e-6,
        weight_decay=WEIGHT_DECAY,
    )
    if scheduler:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max = T_MAX, 
            eta_min = ETA_MIN, 
            verbose = True
        )
    else:
        scheduler = None

    # training and validation
    val_losses = torch.zeros(epochs)
    val_acc = {}
    for metric in METRICS:
        val_acc[metric]=[]

    prev_loss = float('inf')
    no_gain = 0
    training_steps = 0

    last_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        logger.info(f'EPOCH {epoch+1} out of {epochs}\t\t Actual best loss: {prev_loss}')

        val_epoch_loss, val_epoch_metrics = one_epoch(
            model, criterion, optimizer, train_loader, val_loader, device)

        # store the validation metrics
        val_losses[epoch] = val_epoch_loss

         # Store the mean accuracy in each epoch
        for metric in METRICS:
            val_acc[metric].append(val_epoch_metrics[metric])

        # Print metrics
        print_metrics(val_epoch_metrics)
        logger.info(f'Current loss: {val_epoch_loss}')
        update_telegram(bot, "EPOCH {}/{}:\nLoss: {:.4f}".format(epoch+1, epochs, val_epoch_loss))

        # Early stopping management
        if val_epoch_loss < prev_loss:
            prev_loss = val_epoch_loss
            no_gain = 0  #  Resetting early stopping counter
            checkpoint_save(experiment_name, model, epoch + 1)  # saving best model (+1 because the variable starts from 0)
        else:
            no_gain += 1

        if no_gain >= EARLY_STOPPING_PATIENCE:
            logger.info("Quitting training due to early stopping...")
            break

        if scheduler is not None:
            scheduler.step()

        last_epoch = epoch
        training_steps += 1
    
    log_weights, log_loss = None, None

    if training_steps == 0:
        # No training done
        logger.error("No training performed")
        update_telegram(bot, "No training performed")
        return None, None, None, None
    return val_losses[:last_epoch], val_acc, log_weights, log_loss

def main():
    args = parse_args()

    global logger
    logger = get_logger()
    set_level(args.severity)

    train_img_root = args.data + "/train/training_set"
    val_img_root = args.data + "/val/validation_set"

    train_label = args.data + "/train/training_set.txt"
    val_label = args.data + "/val/validation_set.txt"

    experiment_name = EXP_BASE_NAME + " " + strftime("%d %b %H %M", localtime())

    logger.info("Using main device: " + args.device)
    logger.info("Training the backbone" if args.train_bb else "Training only head")
    logger.info("Using {} as attention module".format(args.attention))

    logger.info("Checkpoints will be saved in: " + experiment_name)
    logger.info("Using loss function: " + args.loss)
    logger.info("Using scheduler" if args.sched else "Not using scheduler")
    logger.info("Received the following hyperparameters:")
    logger.info("\tNumber of workers: " + str(args.nw))
    logger.info("\tBatch size: " + str(args.bs))
    logger.info("\tLearning rate: " + str(args.lr))
    logger.info("\tNumber of epochs: " + str(args.epochs))
    logger.info("\tOptimizer: " + str(OPTIMIZER) if not args.optim else strToOptim[args.optim])
    

    model = ANetwork(NUM_CLASSES, train_bb=args.train_bb, backbone_name=args.backbone, backbone_size=args.backbone_size, attention = args.attention)
    model.train()
    model.to(args.device)
    logger.info("Model created")

    preprocessing = model.get_preprocessing()

    training_set = ImageDataset(
        image_folder = train_img_root,
        label_csv = train_label,
        transform = preprocessing
    )
    validation_set = ImageDataset(
        image_folder = val_img_root,
        label_csv = val_label,
        transform = preprocessing
    )

    train_loader = DataLoader(
        training_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers = args.nw
    )
    val_loader = DataLoader(
        validation_set,
        batch_size = args.bs, 
        shuffle=True, 
        num_workers = args.nw
    )   

    kwargs = {
        'use_weights': args.loss == 'cross_custom',
    }


    criterion = Loss(device = args.device, loss_name = args.loss, **kwargs)
    criterion.to(args.device)
    start_epoch = 0

    if args.checkpoint is not None:
        logger.info("Loading checkpoint {}...".format(args.checkpoint))
        checkpoint_load(model, args.checkpoint)

        # Gatherng the starting epoch from the weights
        try:
            epoch_str = re.findall(r'epoch_\d+', args.checkpoint)[0]
            start_epoch = int(re.findall(r'\d+', epoch_str)[0])
        except:
            raise ValueError("Unable to find starting epoch...\n \
                  Checkpoint file name must comprise the string 'epoch_N' in order to start from epoch N")

    # Show what we have loaded
    logger.info("Training set:\t{} samples".format(len(training_set)))
    logger.info("Validation set:\t{} samples".format(len(validation_set)))

    if args.profile:
        global profiler
        profiler = Profile()

    losses, metrics = train(model,
                                                            start_epoch,
                                                            args.epochs,
                                                            args.lr,
                                                            train_loader,
                                                            val_loader,
                                                            criterion,
                                                            args.device,
                                                            experiment_name,
                                                            scheduler=args.sched)

    # Print both accuracy and loss during training
    if losses is not None and metrics is not None: # None when no training is performed
        plot_results(losses, metrics, experiment_name)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("An error occurred during training")
        logger.error(traceback.format_exc())
        logger.error(e)
        update_telegram(bot, "An error occurred during training:\n" + str(e) + "\n" + traceback.format_exc())
        sys.exit(1)
