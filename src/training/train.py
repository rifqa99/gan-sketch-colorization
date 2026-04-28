import torch
from tqdm import tqdm

# from src.models.generator import UNetGenerator
from src.models.convnext_generator import ConvNeXtV2Generator
from src.models.discriminator import PatchGANDiscriminator
from src.losses.gan_loss import adversarial_loss, pixel_loss

import os
import torchvision.utils as vutils

import json

# save_dir = "/content/drive/MyDrive/gan-sketch-colorization/baseline/outputs"  # Baseline
save_dir = "/content/drive/MyDrive/gan-sketch-colorization/convnextv2/outputs"  # ConvNextV2


history = {
    "epoch": [],
    "G_loss": [],
    "G_adv_loss": [],
    "G_L1_loss": [],
    "D_loss": [],
}


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
    total_G_adv_loss = 0
    total_G_L1_loss = 0
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
        G_L1_raw = pixel_loss(fake_img, target_img)
        G_L1_loss = lambda_L1 * G_L1_raw

        G_loss = G_adv_loss + G_L1_loss

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        total_G_loss += G_loss.item()
        total_G_adv_loss += G_adv_loss.item()
        total_G_L1_loss += G_L1_loss.item()
        total_D_loss += D_loss.item()

    avg_G_loss = total_G_loss / len(train_loader)
    avg_G_adv_loss = total_G_adv_loss / len(train_loader)
    avg_G_L1_loss = total_G_L1_loss / len(train_loader)
    avg_D_loss = total_D_loss / len(train_loader)

    return avg_G_loss, avg_G_adv_loss, avg_G_L1_loss, avg_D_loss


def main():
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    from src.datasets.dataset_loader import Edges2ShoesDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{save_dir}/generated_images", exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    dataset_path = "/content/edges2shoes/train"

    dataset = Edges2ShoesDataset(dataset_path)
    # dataset = Subset(dataset, range(5000))

    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    # generator = UNetGenerator().to(device)
    generator = ConvNeXtV2Generator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(),
                             lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 20

    for epoch in range(num_epochs):
        G_loss, G_adv_loss, G_L1_loss, D_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_G,
            optimizer_D,
            device
        )
        sample_input, _ = next(iter(train_loader))
        sample_input = sample_input.to(device)

        generator.eval()

        with torch.no_grad():
            generated = generator(sample_input)

        generator.train()

        vutils.save_image(
            generated,
            f"{save_dir}/generated_images/full_epoch_{epoch+1}.png",
            normalize=True
        )
        history["epoch"].append(epoch + 1)
        history["G_loss"].append(G_loss)
        history["G_adv_loss"].append(G_adv_loss)
        history["G_L1_loss"].append(G_L1_loss)
        history["D_loss"].append(D_loss)

        with open(f"{save_dir}/training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        torch.save(generator.state_dict(),
                   f"{save_dir}/checkpoints/generator_Full_epoch_{epoch+1}.pth")

        torch.save(discriminator.state_dict(),
                   f"{save_dir}/checkpoints/discriminator_Full_epoch_{epoch+1}.pth")
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"G Loss: {G_loss:.4f} | "
            f"G Adv: {G_adv_loss:.4f} | "
            f"G L1: {G_L1_loss:.4f} | "
            f"D Loss: {D_loss:.4f}"
        )


if __name__ == "__main__":
    main()
