import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import lpips

from src.datasets.dataset_loader import Edges2ShoesDataset
from src.models.generator import UNetGenerator


def evaluate_lpips_epoch(epoch, device, loader, lpips_fn, checkpoint_dir):
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"generator_Full_epoch_{epoch}.pth"
    )

    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    total_lpips = 0.0

    with torch.no_grad():
        for input_img, target_img in tqdm(loader, desc=f"Epoch {epoch}"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            fake_img = generator(input_img)

            score = lpips_fn(fake_img, target_img)
            total_lpips += score.mean().item()

    avg_lpips = total_lpips / len(loader)
    return avg_lpips


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_path = "data/edges2shoes/val"
    checkpoint_dir = "/content/drive/MyDrive/gan-sketch-colorization/baseline/outputs/checkpoints"
    save_dir = "baseline/evaluation_lpips"

    os.makedirs(save_dir, exist_ok=True)

    dataset = Edges2ShoesDataset(dataset_path)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    results = {
        "epoch": [],
        "LPIPS": []
    }

    for epoch in range(1, 21):
        print(f"\nEvaluating LPIPS for epoch {epoch}")

        avg_lpips = evaluate_lpips_epoch(
            epoch=epoch,
            device=device,
            loader=loader,
            lpips_fn=lpips_fn,
            checkpoint_dir=checkpoint_dir
        )

        results["epoch"].append(epoch)
        results["LPIPS"].append(avg_lpips)

        print(f"Epoch {epoch} | LPIPS: {avg_lpips:.4f}")

        with open(os.path.join(save_dir, "lpips_baseline_all_epochs.json"), "w") as f:
            json.dump(results, f, indent=4)

    print("\nFinal LPIPS Results:")
    print(results)


if __name__ == "__main__":
    main()
