import matplotlib.pyplot as plt
import asyncio

from statistics import mean
from src.constants import METRICS, BOT_TOKEN
from utils.logger import get_logger
from utils.telegram_bot import TelegramBot, update_telegram

bot = TelegramBot(token=BOT_TOKEN)
logger = get_logger()

def plot_results(loss, metrics, experiment_name):
    # Plot loss during training
    update_telegram(bot, "Final results for experiment {}".format(experiment_name))
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss during training')
    plt.savefig('../{}/loss.jpg'.format(experiment_name))
    asyncio.run(bot.send_photo('../{}/loss.jpg'.format(experiment_name)))
    

    # Plot each metric during training
    plt.figure(figsize=(10, 5))
    for metric in METRICS:
        plt.plot(metrics[metric])
        plt.xlabel('Epochs')
        plt.ylabel(metric)
  
    plt.savefig('../{}/metrics.jpg'.format(experiment_name))
    asyncio.run(bot.send_photo('../{}/metrics.jpg'.format(experiment_name)))

def print_metrics(val_metrics):
    metric_names=METRICS
    metrics={name:[] for name in metric_names}

    metric_str=''
    for name in metric_names:
        metric_str+='\t{} = {:.3f}'.format(name,val_metrics[name])
    logger.info("Metrics:\n {}".format(metric_str))
    update_telegram(bot, "Metrics:\n {}".format(metric_str))
    
    for name in metric_names:
        metrics[name].append(val_metrics[name])
        
    for name in metric_names:
        logger.info("Average {}: {:.5f}".format(name,mean(metrics[name])))
        update_telegram(bot, "Average {}: {:.5f}".format(name,mean(metrics[name])))
