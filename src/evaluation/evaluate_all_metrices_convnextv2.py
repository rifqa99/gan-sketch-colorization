import os
import json
import subprocess
import torch
import lpips

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.datasets.dataset_loader import Edges2ShoesDataset

# IMPORTANT:
# Use the SAME generator class that you used during ConvNeXt V2 training.
# Change this import if your file/class name is different.
from src.models.convnext_generator import ConvNeXtV2Generator


def compute_fid(real_dir, generated_dir):
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", real_dir, generated_dir],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip() + "\n" + result.stderr.strip()
    print(output)

    for line in output.split("\n"):
        if "FID:" in line:
            return float(line.split("FID:")[-1].strip())

    return None


def evaluate_epoch(epoch, device, loader, ssim_metric, lpips_metric, save_dir):
    checkpoint_path = (
        f"/content/drive/MyDrive/gan-sketch-colorization/"
        f"convnextv2/outputs/checkpoints/generator_Full_epoch_{epoch}.pth"
    )

    generated_dir = f"{save_dir}/epoch_{epoch}/generated"
    real_dir = f"{save_dir}/epoch_{epoch}/real"

    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    generator = ConvNeXtV2Generator().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)

    generator.eval()

    total_ssim = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        for idx, (input_img, target_img) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}")
        ):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            fake_img = generator(input_img)

            ssim_score = ssim_metric(fake_img, target_img)
            total_ssim += ssim_score.item()

            lpips_score = lpips_metric(fake_img, target_img)
            total_lpips += lpips_score.mean().item()

            save_image(fake_img, f"{generated_dir}/{idx}.png", normalize=True)
            save_image(target_img, f"{real_dir}/{idx}.png", normalize=True)

    avg_ssim = total_ssim / len(loader)
    avg_lpips = total_lpips / len(loader)

    fid_score = compute_fid(real_dir, generated_dir)

    return avg_ssim, fid_score, avg_lpips


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = "/content/edges2shoes/val"
    save_dir = "/content/drive/MyDrive/gan-sketch-colorization/convnextv2/evaluation_all_epochs"

    os.makedirs(save_dir, exist_ok=True)

    dataset = Edges2ShoesDataset(dataset_path)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    lpips_metric = lpips.LPIPS(net="alex").to(device)
    lpips_metric.eval()

    results = {
        "epoch": [],
        "SSIM": [],
        "FID": [],
        "LPIPS": []
    }

    for epoch in range(1, 21):
        print(f"\nEvaluating epoch {epoch}")

        avg_ssim, fid_score, avg_lpips = evaluate_epoch(
            epoch,
            device,
            loader,
            ssim_metric,
            lpips_metric,
            save_dir
        )

        results["epoch"].append(epoch)
        results["SSIM"].append(avg_ssim)
        results["FID"].append(fid_score)
        results["LPIPS"].append(avg_lpips)

        with open(f"{save_dir}/metrics_all_epochs.json", "w") as f:
            json.dump(results, f, indent=4)

        fid_text = f"{fid_score:.4f}" if fid_score is not None else "None"

        print(
            f"Epoch {epoch} | "
            f"SSIM: {avg_ssim:.4f} | "
            f"FID: {fid_text} | "
            f"LPIPS: {avg_lpips:.4f}"
        )

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()
