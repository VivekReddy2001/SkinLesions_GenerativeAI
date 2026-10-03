"""Sample synthetic images from a trained generator checkpoint.

    python generate.py --checkpoint checkpoints/generator_best.pt --out synthetic/DF --n 200
"""

import argparse
import os

import numpy as np
import torch
from torchvision.utils import save_image

from model.generator import Generator

image_size = (3, 128, 128)
n_noise_features = 100

# For de-normalisation back to [0, 1]
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate synthetic lesion images with a trained WGAN-GP generator.")
    p.add_argument("--checkpoint", required=True, help="generator state dict (.pt)")
    p.add_argument("--out", required=True, help="folder to write PNGs into")
    p.add_argument("--n", type=int, default=200, help="number of images")
    p.add_argument("--filters", type=int, default=128, help="generator filters used in training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    mean_t = torch.FloatTensor(mean).view(3, 1, 1).to(args.device)
    std_t = torch.FloatTensor(std).view(3, 1, 1).to(args.device)

    generator = Generator(n_noise_features, image_size[0], args.filters).to(args.device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    generator.eval()

    noise = torch.from_numpy(np.random.randn(args.n, n_noise_features)).float().to(args.device)
    with torch.no_grad():
        images = generator(noise) * std_t + mean_t
    for j, img in enumerate(images):
        save_image(img, os.path.join(args.out, f"gen_img{j}.png"))
    print(f"wrote {args.n} images to {args.out}")


if __name__ == "__main__":
    main()
