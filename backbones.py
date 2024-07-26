import os
import h5py
import psutil

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_all import dataset_info

openclip_dict = {
    "R50-openai": ("RN50", "openai"),  # 59.6
    "B32-best": ("ViT-B-32", "laion2b_s34b_b79k"),  # 70.2
    "L14-openai": ("ViT-L-14", "openai"),  # 75.3
    # "L14-336-openai": ("ViT-L-14-336", "openai"), # 76.2
    "L14-best": ("ViT-L-14", "datacomp_xl_s13b_b90k"),  # 79.2
    # "G14-best": ("ViT-bigG-14", "laion2b_s39b_b160k"), # 80.1
}

"""
Dataset interface specification
    dataset_name maps to everything, specified in dataset_info object
    dataset_obj has arguments "split" and "transform"
    dataset returns three things: x, y, d
    For Waterbirds etc, groups are determined by (y, d)
    We save all info for each dataset/backbone/split into a single h5 file
"""


def print_memory_usage():
    """Helper function to track memory leakage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")


def get_model(backbone_name):
    # https://github.com/mlfoundations/open_clip#pretrained-model-interface
    arch, pretrained = openclip_dict[backbone_name]
    full_model, _, transform = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    backbone = full_model.visual
    backbone.requires_grad_(False)
    return backbone, transform


def _save_outputs(filename, backbone, loader):
    print(f"Split not found; generating {filename}...")
    # features, labels, domains = [], [], []
    backbone.cuda()
    backbone.eval()

    dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
    output = backbone(dummy_input)
    _, feature_dim = output.shape
    N = len(loader.dataset)
    del dummy_input, output

    temp_fn = filename + ".tmp"
    f = h5py.File(temp_fn, "w")
    features_dset = f.create_dataset("features", (N, feature_dim), dtype="f")
    labels_dset = f.create_dataset("labels", (N,), dtype="i")
    domains_dset = f.create_dataset("domains", (N,), dtype="i")

    i = 0
    for outputs in tqdm(loader):
        if len(outputs) == 2:
            x, y = outputs
            y = y.numpy()
            d = np.zeros_like(y)
        elif len(outputs) >= 3:
            x, y, d, *_ = outputs
            y, d = y.numpy(), d.numpy()

        x = x.cuda()
        with torch.no_grad():
            _features = backbone(x).detach().cpu().numpy()

        features_dset[i : i + len(_features)] = _features
        labels_dset[i : i + len(_features)] = y
        domains_dset[i : i + len(_features)] = d
        i += len(_features)
        del x, y, d, outputs, _features
        print_memory_usage()

    filenames = loader.dataset.filename_array
    if isinstance(filenames, np.ndarray):
        filenames = filenames.tolist()
    f.create_dataset("filenames", data=filenames)
    f.close()
    os.rename(temp_fn, filename)


def read_h5(filename):
    print(f"Reading {filename}")
    with h5py.File(filename, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        domains = f["domains"][:]
        filenames = f["filenames"][:]
    info_dict = {"features": features, "labels": labels, "domains": domains, "filenames": filenames}
    return info_dict


def get_outputs(dataset_name, backbone_name, split):
    assert dataset_name in dataset_info.keys(), f"{dataset_name} not found"
    os.makedirs("features", exist_ok=True)

    # outputs_fn = f"features/{dataset_name}-{backbone_name}-{split}.pkl"
    outputs_fn = f"features/{dataset_name}-{backbone_name}-{split}.h5"
    if os.path.exists(outputs_fn):
        return read_h5(outputs_fn)

    backbone, transform = get_model(backbone_name)

    data_obj = dataset_info[dataset_name]["data_obj"]
    dataset = data_obj(split=split, transform=transform)
    print(f"{dataset_name} split {split} size: {len(dataset)}")
    print(f"Transformed image size: {dataset[0][0].shape} (should be 3x224x224)")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    _save_outputs(outputs_fn, backbone, loader)
    return read_h5(outputs_fn)


def feedforward_text(prompts, backbone_name):
    arch, pretrained = openclip_dict[backbone_name]
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    text_features = []
    tokenized_features = tokenizer(prompts)
    text_features = model.encode_text(tokenized_features)
    text_features = text_features.detach().cpu().numpy()
    return text_features


if __name__ == "__main__":
    from itertools import product

    all_datasets = dataset_info.keys()
    all_backbones = ["R50-openai", "L14-openai", "L14-best"]
    splits = ["train", "val", "test"]
    for dataset, backbone, split in product(all_datasets, all_backbones, splits):
        outputs_dict = get_outputs(dataset, backbone, split)
        for name, arr in outputs_dict.items():
            print(name, arr.shape)
