import torch
import torch.nn as nn
import sys
sys.path.extend([".", ".."])

from torchvision import transforms as t
from torchvision.models import convnext_base, convnext_large, convnext_small, convnext_tiny, ConvNeXt_Base_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Large_Weights, swin_v2_t, swin_v2_s, swin_v2_b, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from collections import OrderedDict
from itertools import chain

from models.cbam import CBAM
from models.ham import HAM
from models.mlp import MLP
from utils.logger import get_logger


logger = get_logger()

def get_basenet(backbone_name, backbone_size):
    """
    Returns a model of the specified type with pretrained weights.

    Parameters
    ----------
    backbone_name : str
        The available backbone names are ["convnext", "swin", "resnext", "resnet"]

    backbone_size : str
        The available backbone sizes are:
        ["tiny", "small", "base", "large"] if the backbone is "convnext";
        ["tiny", "small", "base"] if the backbone is "swin";
        ["small", "base", "large"] if the backbone is "resnext"
        ["nano", "tiny", "small", "base", "large"] if the backbone is "resnet"
    """
    if backbone_name == "convnext":
        if backbone_size == "tiny":
            base_net = convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        if backbone_size == "small":
            base_net = convnext_small(
                weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        if backbone_size == "base":
            base_net = convnext_base(
                weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        if backbone_size == "large":
            base_net =convnext_large(
                weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    elif backbone_name == "swin":
        if backbone_size == "tiny":
            base_net = swin_v2_t(weights="DEFAULT")
        if backbone_size == "small":
            base_net = swin_v2_s(weights="DEFAULT")
        if backbone_size == "base":
            base_net = swin_v2_b(weights="DEFAULT")
    elif backbone_name == "resnext":
        if backbone_size == "small":
            base_net = resnext50_32x4d(weights="DEFAULT")
        if backbone_size == "base":
            base_net = resnext101_32x8d(weights="DEFAULT")
        if backbone_size == "large":
            base_net = resnext101_64x4d(weights="DEFAULT")
    elif backbone_name == "resnet":
        if backbone_size == "nano":
            base_net = resnet18(weights="DEFAULT")
        if backbone_size == "tiny":
            base_net = resnet34(weights="DEFAULT")
        if backbone_size == "small":
            base_net = resnet50(weights="DEFAULT")
        if backbone_size == "base":
            base_net = resnet101(weights="DEFAULT")
        if backbone_size == "large":
            base_net = resnet152(weights="DEFAULT")

    return base_net

class ANetwork(nn.Module):
    # The order for the outputs is: upper_color, lower_color, gender, bag, hat
    FINAL_RES = (192,72) # (h, w) -> ratio 2.6:1
    
    def __init__(self, num_classes, train_bb = True, backbone_name = "convnext", backbone_size = "base", attention = "cbam", xavier = False, linear_layers = 3):
        """
        Initializes the model

        Parameters
        ----------
        train_bb : bool
            If True, the backbone will be entirely trained, otherwise it will be frozen

        backbone_name : str
            The available backbone names are ["convnext", "swin", "resnext", "resnet"]

        backbone_size : str
            The available backbone sizes are:
            ["tiny", "small", "base", "large"] if the backbone is "convnext";
            ["tiny", "small", "base"] if the backbone is "swin";
            ["small", "base", "large"] if the backbone is "resnext"
            ["nano", "tiny", "small", "base", "large"] if the backbone is "resnet"

        attention : str
            The attention mechanism to use, can be one of ["cbam", "ham", "none"]

        xavier : bool
            If True, the weights of the classification heads will be initialized with Xavier normal
        """
        if not isinstance(train_bb, bool):
            raise ValueError("The train_bb parameter must be a boolean")
        
        if not isinstance(backbone_name, str):
            raise ValueError("The backbone type type must be a string")
                
        if backbone_name not in ["convnext", "swin", "resnext", "resnet"]:
            raise ValueError("The backbone name must be one of ['resnet', 'resnext', 'convnext', 'swin']")
        
        if not isinstance(backbone_size, str):
            raise ValueError("The backbone size must be a string")
        
        if not isinstance(attention, str):
            raise ValueError("The attention mechanism must be a string")
        
        if attention not in ["cbam", "ham", "none"]:
            raise ValueError("The attention mechanism must be one of ['cbam', 'ham', 'none']")
        
        if backbone_name == "convnext" and backbone_size not in ["tiny", "small", "base", "large"]:
            raise ValueError("The convnext size must be one of ['tiny', 'small', 'base', 'large']")
        
        if backbone_name == "swin" and backbone_size not in ["tiny", "small", "base"]:
            raise ValueError("The swin size must be one of ['tiny', 'small', 'base']")

        if backbone_name == "resnext" and backbone_size not in ["small", "base", "large"]:
            raise ValueError("The resnext size must be one of ['small', 'base', 'large']")

        if backbone_name == "resnet" and backbone_size not in ["nano", "tiny", "small", "base", "large"]:
            raise ValueError("The resnet size must be one of ['nano', 'tiny', 'small', 'base', 'large']")

        super(ANetwork, self).__init__()
    
        # Load the base network from PyTorch
        self.base_net = get_basenet(backbone_name, backbone_size)

        # Extract backbone from basenet
        if backbone_name == "convnext":
            self.backbone = nn.Sequential(*list(self.base_net.children())[:-2])
        elif backbone_name == "swin":
            self.backbone = nn.Sequential(*list(self.base_net.children())[:-3])
        elif backbone_name == "resnext":
            self.backbone = nn.Sequential(*list(self.base_net.children())[:-2])
        elif backbone_name == "resnet":
            self.backbone = nn.Sequential(*list(self.base_net.children())[:-2])

        self.backbone_name = backbone_name
        self.backbone_size = backbone_size
        features_shape = self.__calculate_output_shape()


        self.train_bb = train_bb
        if not self.train_bb:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feature_maps = features_shape[1]


        self.ATT = nn.Identity
        if attention == "cbam":
            self.ATT = CBAM
        elif attention == "ham":
            self.ATT = HAM

        att_maps = self.__calculate_att_output_shape()

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(OrderedDict([('att', self.ATT(feature_maps)), ('flatten', self.flatten), ('classifier', MLP(att_maps, num_classes, num_layers = linear_layers, dropout = [0,1], dynamic_creation = (self.backbone_name == "resnext")))]))

        if xavier:
            def initialize_weights(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    logger.debug("Applying xavier normal on {}".format(m))
                    nn.init.xavier_normal_(m.weight, gain=1)
                    # if m.bias is not None:
                    #     nn.init.constant_(m.bias, 0)

            # Perform Xavier initialization on classification heads
            for h in self.heads:
                h.apply(initialize_weights)

        self.device = None
        self.eval()
        logger.info("Network initialized with {}-{} backbone and {} module".format(backbone_name, backbone_size, attention))

    def __calculate_att_output_shape(self):
        # Utility method to calculate the output shape of the attention mechanism
        features_shape = self.__calculate_output_shape()

        example_input = torch.randn(features_shape)

        att = self.ATT(features_shape[1])

        output_feature = att(example_input)
        flattened_out = nn.Flatten()(output_feature)
        return flattened_out.shape[1]


    def __calculate_output_shape(self):
        # Utility method to calculate the output shape of the backbone
        example_input = torch.randn((2, 3, self.FINAL_RES[0], self.FINAL_RES[1]))  # (batch_size, channels, height, width)
        output_feature = self.backbone(example_input)
        return output_feature.shape

    def forward(self, x):
        """
        Forward pass of the PedNet model

        Parameters
        ----------
        x : torch.Tensor
            The input tensor, with shape (batch_size, channels, height, width) or (channels, height, width)

        Returns
        -------
        out : list
            A list of tensors, each one representing the output of a head. If single task training, the list will contain only one element different from None.
        """
        features = self.backbone(x)

        ho = self.head(features)

        # Must be BxC
        if len(ho.shape) == 1:
            ho = ho.unsqueeze(0)

        return ho
    
    def to(self, device):
        """
        Overrides the to method in order to set the device for the backbone and the heads
        
        Parameters
        ----------
        device : str
            The device to use, can be one of ['cuda', 'cpu', 'mps']

        Raises
        ------
        ValueError
            If the device is not one of ['cuda', 'cpu', 'mps']        
        """
        if device == 'cuda':
            self.cuda()
        elif device == 'cpu':
            self.cpu()
        elif device == 'mps':
            self.mps()
        else:
            raise ValueError("The device must be one of ['cuda', 'cpu', 'mps']")

    def cuda(self):
        """
        Overrides the cuda method in order to set the device for the backbone and the heads
        """
        self.device = 'cuda'
        self.backbone.cuda()
        self.head.cuda()

    def cpu(self):
        """
        Overrides the cpu method in order to set the device for the backbone and the heads
        """
        self.device = 'cpu'
        self.backbone.cpu()
        self.head.cpu()

    def mps(self):
        """
        Overrides the mps method in order to set the device for the backbone and the heads
        """
        self.device = 'mps'
        self.backbone.to('mps')
        self.head.to('mps')

    def all_parameters(self):
        """
        Returns an iterator of all the net parameters

        Returns
        -------
        param : iterator
            An iterator of all the net parameters
        """
        param = self.backbone.parameters()

        param = chain(param, self.head.parameters())

        return iter(param)
    
    def parameters(self):
        """
        Returns an iterator of all the net parameters that require optimization.

        Returns
        -------
        param : iterator
            An iterator of all the net parameters that require optimization
        """
        param = iter([])

        # In this way we can dynamically manage the backbone being trainable or not
        bbone_par = self.backbone.parameters()
        if self.train_bb:
            param=chain(param,bbone_par)

        param = chain(param, self.head.parameters())

        return iter(param)

    def eval(self):
        """
        Overrides the eval method in order to set the backbone and the heads in evaluation mode
        """
        self.backbone.eval()
        self.head.eval()

    def train(self):
        """
        Overrides the train method in order to set the backbone and the heads in training mode
        """
        self.backbone.train()
        self.head.train()

    def get_preprocessing(self):
        """
        Returns the preprocessing pipeline for the PedNet model
        """
        return t.Compose([t.ToTensor(), # Automatically converted in range [0,1]
                          t.Resize(self.FINAL_RES, antialias=True),
                          t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ])

    def __str__(self):
        """
        Returns a string representation of the model

        Returns
        -------
        str
            A string representation of the model
        """
        return str(self.backbone) + "\n" + str(self.head)

    def get_last_shared_layer(self):
        """
        Returns the last shared layer. Used by GradNorm to optimize loss weights.

        Returns
        -------
        layer : nn.Module
            The last shared layer
        """

        # Considering only the weighted layers
        if self.backbone_name == "convnext":  
            return self.backbone[-1][-1][-1].block[5]
        elif self.backbone_name == "swin":
            return self.backbone[0][-1][-1].mlp[-2]
        elif self.backbone_name == "resnext":
            return self.backbone[-1][-1].conv3
        elif self.backbone_name == "resnet":
            if self.backbone_size == "nano":
                return self.backbone[-1][-1].conv2
            return self.backbone[-1][-1].conv3

    def num_parameters(self):
        """
        Returns the number of parameters of the PedNet model

        Returns
        -------
        int
            The number of parameters of the PedNet model
        """
        return sum(p.numel() for p in self.all_parameters())

if __name__ == '__main__':
    net = ANetwork(backbone_name="convnext", backbone_size = "base", attention="cbam")
    print(net)
    print(net.num_parameters())
    net.eval()
    net.cuda()

    x = torch.randn((1, 3, net.FINAL_RES[0], net.FINAL_RES[1])).cuda()
    print(net(x))

    import time
    time.sleep(10)