import os
import torch

def checkpoint_save(experiment_name, model, epoch):
    """
    Save the model's state_dict to a file.

    Parameters:
    ----------
    experiment_name : str
        The name of the experiment.

    model : torch.nn.Module
        The model to save.

    epoch : int
        The epoch number.
    """
    save_path = "../{}/checkpoints/".format(experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, "epoch_{}.pth".format(epoch))
    torch.save({
        'backbone_state_dict': model.backbone.state_dict(),
        'head_state_dict': model.head.state_dict(),
    }, save_path)

def checkpoint_load(model, path):
    """
    Load the model's state_dict from a file.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to load the state_dict into.

    path : str
        The path to the file.
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.head.load_state_dict(checkpoint['head_state_dict'])
