import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.datasets.dataset_loader import Edges2ShoesDataset
from src.models.generator import UNetGenerator


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "/content/drive/MyDrive/gan-sketch-colorization/baseline/outputs/checkpoints/generator_Full_epoch_16.pth"

    dataset_path = "/content/edges2shoes/val"

    save_dir = "/content/drive/MyDrive/gan-sketch-colorization/baseline/evaluation"

    os.makedirs(f"{save_dir}/generated", exist_ok=True)
    os.makedirs(f"{save_dir}/real", exist_ok=True)

    dataset = Edges2ShoesDataset(dataset_path)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    total_ssim = 0

    with torch.no_grad():

        for idx, (input_img, target_img) in enumerate(tqdm(loader)):

            input_img = input_img.to(device)
            target_img = target_img.to(device)

            fake_img = generator(input_img)

            ssim_score = ssim_metric(fake_img, target_img)

            total_ssim += ssim_score.item()

            save_image(
                fake_img,
                f"{save_dir}/generated/{idx}.png",
                normalize=True
            )

            save_image(
                target_img,
                f"{save_dir}/real/{idx}.png",
                normalize=True
            )

    avg_ssim = total_ssim / len(loader)

    print(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    main()
