"""Train a WGAN-GP on one lesion class, as used for the SkinAid paper.

One generator is trained per class, from a folder holding that class's
images (for example all dermatofibroma images of HAM10000):

    python wgan_gp.py --data data/DF --epochs 10000

Checkpoints go to --out (default: checkpoints/), sample grids to
training/ and loss curves to plots/. The training procedure is unchanged
from the paper: 5 critic steps per generator step (100 for the first 25
generator iterations and every 500th), gradient penalty λ = 10, Adam with
lr 1e-4 and β = (0, 0.9), batch 32, 128×128 RGB images normalised to [-1, 1].
"""

import argparse
import os
import time

import numpy as np
import torch
import torchvision.transforms as tt
import torchvision.utils as vutils

from dataset import Dataset
from model.discriminator import Discriminator
from model.generator import Generator
from utils import compute_gradient_penalty, penalty, plot, w_distance

# Hyper-parameters used for the paper
image_size = (3, 128, 128)
n_noise_features = 100  # noise vector dimension
disc_steps_default = 5
lambda_pen = 10
discriminator_filters = 128
generator_filters = 128

# mean & standard deviation for normalisation and de-normalisation
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder of training images for one lesion class")
    p.add_argument("--out", default="checkpoints", help="where generator/critic checkpoints are written")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=500, help="save numbered checkpoints every N epochs")
    p.add_argument("--sample-every", type=int, default=100, help="save a sample grid every N epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    for d in (args.out, "plots", "training"):
        os.makedirs(d, exist_ok=True)

    mean_t = torch.FloatTensor(mean).view(3, 1, 1).to(device)
    std_t = torch.FloatTensor(std).view(3, 1, 1).to(device)

    transform = tt.Compose([tt.Resize((128, 128)), tt.ToTensor(), tt.Normalize(mean, std)])
    dset_train = Dataset(args.data, transform)
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)

    discriminator = Discriminator(image_size[0], discriminator_filters).to(device)
    generator = Generator(n_noise_features, image_size[0], generator_filters).to(device)

    def checkpoint(epoch):
        torch.save(discriminator.state_dict(), os.path.join(args.out, f"discriminator_{epoch}.pt"))
        torch.save(generator.state_dict(), os.path.join(args.out, f"generator_{epoch}.pt"))

    def generate_sample(epoch, frame_noise):
        """Save random samples, plus samples from a fixed noise vector so progress is comparable."""
        random_noise = torch.from_numpy(np.random.randn(args.batch_size, n_noise_features)).float().to(device)
        with torch.no_grad():
            gen_output_random = generator(random_noise) * std_t + mean_t
            gen_output_frame = generator(frame_noise) * std_t + mean_t
        vutils.save_image(vutils.make_grid(gen_output_random.cpu()[:4]), f"training/img_generator_epoch_{epoch}.png")
        vutils.save_image(vutils.make_grid(gen_output_frame.cpu()), "training/fixed_noise_latest.png")

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))

    disc_losses, gen_losses, w_distances, gradient_penalty_list = [], [], [], []
    gen_iterations = 0

    # A fixed noise vector, to check progress visually across epochs.
    frame_noise = torch.from_numpy(np.random.randn(args.batch_size, n_noise_features)).float().to(device)

    gen_min_loss = float("inf")
    for e in range(args.epochs):
        start = time.time()
        epoch_dlosses, epoch_glosses = [], []
        train_iterator = iter(train_loader)
        i = 0
        while i < len(train_loader):
            # ---------------- train the critic ----------------
            for p in discriminator.parameters():
                p.requires_grad = True
            # extra critic steps early on and periodically, as in the WGAN paper
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                disc_steps = 100
            else:
                disc_steps = disc_steps_default
            j = 0
            step_loss = 0.0
            while j < disc_steps and i < len(train_loader):
                j += 1
                i += 1
                images = next(train_iterator).to(device)
                common_batch_size = min(args.batch_size, images.shape[0])
                disc_optimizer.zero_grad()
                noises = torch.from_numpy(np.random.randn(common_batch_size, n_noise_features)).float().to(device)
                disc_output = discriminator(images)
                gen_images = generator(noises)
                gen_output = discriminator(gen_images)

                gradient_penalty = compute_gradient_penalty(images, gen_images, discriminator, lambda_pen)
                loss = torch.mean(gen_output - disc_output + gradient_penalty)
                loss.backward()
                wdist = torch.mean(disc_output - gen_output)
                disc_optimizer.step()

                step_loss += loss.item()
                epoch_dlosses.append(loss.item())
                w_distances.append(wdist.item())
                gradient_penalty_list.append(torch.mean(gradient_penalty).item())
            disc_losses.append(step_loss / j)

            # ---------------- train the generator ----------------
            for p in discriminator.parameters():
                p.requires_grad = False
            gen_optimizer.zero_grad()
            noises = torch.from_numpy(np.random.randn(args.batch_size, n_noise_features)).float().to(device)
            loss = -torch.mean(discriminator(generator(noises)))
            loss.backward()
            gen_optimizer.step()
            gen_losses.append(loss.item())
            # keep the checkpoint with the lowest non-negative generator loss
            if gen_min_loss >= loss.item() >= 0.0:
                gen_min_loss = loss.item()
                checkpoint("best")
            epoch_glosses.append(loss.item())
            gen_iterations += 1

        print(
            f"Epoch {e}  D loss: {np.mean(epoch_dlosses):.5f}  G loss: {np.mean(epoch_glosses):.5f}  "
            f"Time: {time.time() - start:.0f}s"
        )
        if e % args.sample_every == 0:
            generate_sample(e, frame_noise)
        if e % args.checkpoint_every == 0:
            checkpoint(e)
        plot(disc_losses, gen_losses)
        penalty(gradient_penalty_list)
        w_distance(w_distances)

    checkpoint("final")


if __name__ == "__main__":
    main()
