"""
Latent factor values
Color: white
Shape: square, ellipse, heart
Scale: 6 values linearly spaced in [0.5, 1]
Orientation: 40 values in [0, 2 pi]
Position X: 32 values in [0, 1]
Position Y: 32 values in [0, 1]
"""

import os

import numpy as np
from matplotlib import pyplot as plt


def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)

    return samples


def sample_colored_images(shape, color):
    N = 500
    assert shape in [0, 1, 2]
    assert color in [0, 1, 2]
    latent_factors = sample_latent(size=N)
    latent_factors[:, 1] = shape  # Set shape factor to 1 (square)
    latent_factors[:, 2] = np.random.randint(2, 6, size=N)

    # Convert latent factors to indices
    indices = latent_to_index(latent_factors)

    # Get the corresponding square images
    images = imgs[indices] * 1.0
    colored_images = np.stack((images, images, images), axis=3)
    for i in range(3):
        if i != color:
            colored_images[:, :, :, i] = 0

    # Display the sampled squares
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(colored_images[i])
        ax.axis("off")
    plt.show()

    shape_name = ["square", "ellipse", "heart"][shape]
    color_name = ["red", "green", "blue"][color]
    save_dir = f"data/colored_dsprites/{shape_name}-{color_name}"
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(colored_images):
        fn = f"{save_dir}/{i}.png"
        plt.imsave(fn, img)


if __name__ == "__main__":
    dataset_zip = np.load(
        "data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", allow_pickle=True, encoding="latin1"
    )
    print("Keys in the dataset:", dataset_zip.keys())
    imgs = dataset_zip["imgs"]
    latents_values = dataset_zip["latents_values"]
    latents_classes = dataset_zip["latents_classes"]
    metadata = dataset_zip["metadata"][()]

    latents_sizes = metadata["latents_sizes"]
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1])))

    sample_colored_images(0, 0)
    sample_colored_images(0, 2)
    sample_colored_images(1, 0)
    sample_colored_images(1, 2)
