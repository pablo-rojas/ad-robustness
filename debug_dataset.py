#!/usr/bin/env python
import argparse
import torch
from src.dataset_utils import get_dataset, get_loaders

def main():
    parser = argparse.ArgumentParser(description="Debug dataset image statistics before and after normalization")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar", "imagenet"],
        required=True,
        help="Dataset to load (mnist, cifar, imagenet)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for data loader"
    )
    parser.add_argument(
        "--num_batches", type=int, default=10, help="Number of batches to process"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of data loader workers"
    )
    args = parser.parse_args()

    # Get the dataset and loaders.
    dataset = get_dataset(args.dataset)
    train_loader, _ = get_loaders(dataset, workers=args.workers, batch_size=args.batch_size)
    norm = dataset.normalize  # normalization transformation defined in your dataset

    # Initialize accumulators for original images.
    tot_sum_orig = 0.0
    tot_sum_sq_orig = 0.0
    tot_pixels_orig = 0
    global_min_orig = None
    global_max_orig = None

    # And for normalized images.
    tot_sum_norm = 0.0
    tot_sum_sq_norm = 0.0
    tot_pixels_norm = 0
    global_min_norm = None
    global_max_norm = None

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: [batch, channels, height, width]
        # --- Original images stats ---
        batch_min_orig = images.min()
        batch_max_orig = images.max()
        batch_sum_orig = images.sum().item()
        batch_sum_sq_orig = (images ** 2).sum().item()
        count_orig = images.numel()

        if global_min_orig is None or batch_min_orig < global_min_orig:
            global_min_orig = batch_min_orig
        if global_max_orig is None or batch_max_orig > global_max_orig:
            global_max_orig = batch_max_orig

        tot_sum_orig += batch_sum_orig
        tot_sum_sq_orig += batch_sum_sq_orig
        tot_pixels_orig += count_orig

        # --- Normalized images stats ---
        norm_images = norm(images)  # apply the normalization transform
        batch_min_norm = norm_images.min()
        batch_max_norm = norm_images.max()
        batch_sum_norm = norm_images.sum().item()
        batch_sum_sq_norm = (norm_images ** 2).sum().item()
        count_norm = norm_images.numel()

        if global_min_norm is None or batch_min_norm < global_min_norm:
            global_min_norm = batch_min_norm
        if global_max_norm is None or batch_max_norm > global_max_norm:
            global_max_norm = batch_max_norm

        tot_sum_norm += batch_sum_norm
        tot_sum_sq_norm += batch_sum_sq_norm
        tot_pixels_norm += count_norm
        
        if batch_idx + 1 >= args.num_batches:
            break

    # Compute overall statistics for original images.
    mean_orig = tot_sum_orig / tot_pixels_orig
    var_orig = tot_sum_sq_orig / tot_pixels_orig - mean_orig ** 2
    std_orig = var_orig ** 0.5

    # And for normalized images.
    mean_norm = tot_sum_norm / tot_pixels_norm
    var_norm = tot_sum_sq_norm / tot_pixels_norm - mean_norm ** 2
    std_norm = var_norm ** 0.5

    print("\nOverall statistics (computed over {} batches):".format(args.num_batches))
    print("Original images:")
    print(f"  Min: {global_min_orig.item():.4f}")
    print(f"  Max: {global_max_orig.item():.4f}")
    print(f"  Mean: {mean_orig:.4f}")
    print(f"  Std:  {std_orig:.4f}")

    print("\nNormalized images:")
    print(f"  Min: {global_min_norm.item():.4f}")
    print(f"  Max: {global_max_norm.item():.4f}")
    print(f"  Mean: {mean_norm:.4f}")
    print(f"  Std:  {std_norm:.4f}")

if __name__ == "__main__":
    main()
