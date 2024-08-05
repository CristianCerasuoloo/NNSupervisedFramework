import torch 

# Data parameters
NUM_CLASSES = 2

# Training parameters
EARLY_STOPPING_PATIENCE = 10 
EPOCHS = 50
WORKERS = 4
BATCH_SIZE = 96
LEARNING_RATE = 1e-4
EXP_BASE_NAME = "BaseName"
WEIGHT_DECAY = 0.05 # 1e-4
OPTIMIZER = torch.optim.AdamW

# Scheduler parameters
ETA_MIN = LEARNING_RATE * 1e-3
T_MAX = EPOCHS