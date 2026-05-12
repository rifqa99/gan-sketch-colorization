import os
import json
import torch
import torch.optim as optim
import torchvision.utils as vutils

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.convnext_generator import ConvNeXtV2Generator
from src.models.discriminator import PatchGANDiscriminator
from src.losses.gan_loss import adversarial_loss, pixel_loss
from src.datasets.dataset_loader import Edges2ShoesDataset


save_dir = "/content/drive/MyDrive/gan-sketch-colorization/convnextv2_l2_reduced/outputs"


def train_one_epoch(
    generator,
    discriminator,
    train_loader,
    optimizer_G,
    optimizer_D,
    device,
    lambda_L2=100
):
    generator.train()
    discriminator.train()

    total_G_loss = 0
    total_G_adv_loss = 0
    total_G_L2_loss = 0
    total_D_loss = 0

    for input_img, target_img in tqdm(train_loader):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        # Train Discriminator
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

        # Train Generator
        fake_img = generator(input_img)
        fake_pred = discriminator(input_img, fake_img)

        G_adv_loss = adversarial_loss(fake_pred, valid)

        G_L2_raw = pixel_loss(fake_img, target_img)
        G_L2_loss = lambda_L2 * G_L2_raw

        G_loss = G_adv_loss + G_L2_loss

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        total_G_loss += G_loss.item()
        total_G_adv_loss += G_adv_loss.item()
        total_G_L2_loss += G_L2_loss.item()
        total_D_loss += D_loss.item()

    return (
        total_G_loss / len(train_loader),
        total_G_adv_loss / len(train_loader),
        total_G_L2_loss / len(train_loader),
        total_D_loss / len(train_loader),
    )


def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"{save_dir}/generated_images", exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    dataset_path = "/content/edges2shoes/train"
    dataset = Edges2ShoesDataset(dataset_path)

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    generator = ConvNeXtV2Generator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(),
                             lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=0.0002, betas=(0.5, 0.999))

    history = {
        "epoch": [],
        "G_loss": [],
        "G_adv_loss": [],
        "G_L2_loss": [],
        "D_loss": []
    }

    start_epoch = 15
    num_epochs = 30
    resume_training = True

    if resume_training:
        checkpoint_path = f"{save_dir}/checkpoints/full_checkpoint_epoch_{start_epoch}.pth"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)

            generator.load_state_dict(checkpoint["generator_state_dict"])
            discriminator.load_state_dict(
                checkpoint["discriminator_state_dict"])
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

            history = checkpoint.get("history", history)
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed from epoch {checkpoint['epoch']}")
        else:
            print(
                f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    for epoch in range(start_epoch, num_epochs + 1):
        G_loss, G_adv_loss, G_L2_loss, D_loss = train_one_epoch(
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
            f"{save_dir}/generated_images/full_epoch_{epoch}.png",
            normalize=True
        )

        history["epoch"].append(epoch)
        history["G_loss"].append(G_loss)
        history["G_adv_loss"].append(G_adv_loss)
        history["G_L2_loss"].append(G_L2_loss)
        history["D_loss"].append(D_loss)

        with open(f"{save_dir}/training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        torch.save(
            generator.state_dict(),
            f"{save_dir}/checkpoints/generator_Full_epoch_{epoch}.pth"
        )

        torch.save(
            discriminator.state_dict(),
            f"{save_dir}/checkpoints/discriminator_Full_epoch_{epoch}.pth"
        )

        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
            "history": history,
        }, f"{save_dir}/checkpoints/full_checkpoint_epoch_{epoch}.pth")

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"G Loss: {G_loss:.4f} | "
            f"G Adv: {G_adv_loss:.4f} | "
            f"G L2: {G_L2_loss:.4f} | "
            f"D Loss: {D_loss:.4f}"
        )


if __name__ == "__main__":
    main()
