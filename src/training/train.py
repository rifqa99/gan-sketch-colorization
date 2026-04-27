import torch
from tqdm import tqdm

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.losses.gan_loss import adversarial_loss, pixel_loss


def train_one_epoch(
    generator,
    discriminator,
    train_loader,
    optimizer_G,
    optimizer_D,
    device,
    lambda_L1=100
):
    generator.train()
    discriminator.train()

    total_G_loss = 0
    total_D_loss = 0

    for input_img, target_img in tqdm(train_loader):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        # --------------------
        # Train Discriminator
        # --------------------
        fake_img = generator(input_img).detach()

        real_pred = discriminator(input_img, target_img)
        fake_pred = discriminator(input_img, fake_img)

        valid = torch.ones_like(real_pred, device=device)
        fake = torch.zeros_like(fake_pred, device=device)

        real_loss = adversarial_loss(real_pred, valid)
        fake_loss = adversarial_loss(fake_pred, fake)

        D_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # ----------------
        # Train Generator
        # ----------------
        fake_img = generator(input_img)

        fake_pred = discriminator(input_img, fake_img)

        G_adv_loss = adversarial_loss(fake_pred, valid)
        G_L1_loss = pixel_loss(fake_img, target_img)

        G_loss = G_adv_loss + lambda_L1 * G_L1_loss

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        total_G_loss += G_loss.item()
        total_D_loss += D_loss.item()

    avg_G_loss = total_G_loss / len(train_loader)
    avg_D_loss = total_D_loss / len(train_loader)

    return avg_G_loss, avg_D_loss
