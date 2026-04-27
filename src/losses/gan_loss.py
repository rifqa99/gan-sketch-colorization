import torch.nn as nn

adversarial_loss = nn.BCEWithLogitsLoss()
pixel_loss = nn.L1Loss()
