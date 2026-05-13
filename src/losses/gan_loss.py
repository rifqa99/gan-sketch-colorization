import torch.nn as nn

adversarial_loss = nn.BCEWithLogitsLoss()
pixel_loss = nn.MSELoss()
pixel_loss_l1 = nn.L1Loss()
