import os
import json
import subprocess
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.datasets.dataset_loader import Edges2ShoesDataset
from src.models.generator import UNetGenerator


def compute_fid(real_dir, generated_dir):
    result = subprocess.run(
        [
            "python", "-m", "pytorch_fid",
            real_dir,
            generated_dir
        ],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()
    print(output)

    for line in output.split("\n"):
        if "FID:" in line:
            return float(line.split("FID:")[-1].strip())

    return None


def evaluate_epoch(epoch, device, loader, ssim_metric, save_dir):
    checkpoint_path = (
        f"/content/drive/MyDrive/gan-sketch-colorization/"
        f"baseline/outputs/checkpoints/generator_Full_epoch_{epoch}.pth"
    )

    generated_dir = f"{save_dir}/epoch_{epoch}/generated"
    real_dir = f"{save_dir}/epoch_{epoch}/real"

    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    total_ssim = 0

    with torch.no_grad():
        for idx, (input_img, target_img) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            fake_img = generator(input_img)

            ssim_score = ssim_metric(fake_img, target_img)
            total_ssim += ssim_score.item()

            save_image(fake_img, f"{generated_dir}/{idx}.png", normalize=True)
            save_image(target_img, f"{real_dir}/{idx}.png", normalize=True)

    avg_ssim = total_ssim / len(loader)
    fid_score = compute_fid(real_dir, generated_dir)

    return avg_ssim, fid_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = "/content/edges2shoes/val"
    save_dir = "/content/drive/MyDrive/gan-sketch-colorization/baseline/evaluation_all_epochs"

    os.makedirs(save_dir, exist_ok=True)

    dataset = Edges2ShoesDataset(dataset_path)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    results = {
        "epoch": [],
        "SSIM": [],
        "FID": []
    }

    for epoch in range(1, 21):
        print(f"\nEvaluating epoch {epoch}")

        avg_ssim, fid_score = evaluate_epoch(
            epoch,
            device,
            loader,
            ssim_metric,
            save_dir
        )

        results["epoch"].append(epoch)
        results["SSIM"].append(avg_ssim)
        results["FID"].append(fid_score)

        with open(f"{save_dir}/metrics_all_epochs.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"Epoch {epoch} | SSIM: {avg_ssim:.4f} | FID: {fid_score:.4f}")

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()
