import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Edges2ShoesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        _, H, W = image.shape

        input_image = image[:, :, :W // 2]
        target_image = image[:, :, W // 2:]

        return input_image, target_image
