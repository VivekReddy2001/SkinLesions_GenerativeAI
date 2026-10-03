import os

import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class Dataset(torch.utils.data.Dataset):
    """Loads every image in a folder (one lesion class) as an RGB tensor."""

    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        # Skip anything that is not an image (.DS_Store, CSVs, sub-folders).
        self.filename = sorted(f for f in os.listdir(image_path) if f.lower().endswith(IMAGE_EXTENSIONS))
        if not self.filename:
            raise ValueError(f"no images found in {image_path!r}")
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __getitem__(self, index):
        # convert('RGB') so grayscale or RGBA files still give 3 channels.
        image = Image.open(os.path.join(self.image_path, self.filename[index])).convert("RGB")
        return self.transform(image)  # C x H x W

    def __len__(self):
        return len(self.filename)
