# https://github.com/AndPotap/afr/blob/main/data/datasets.py
# https://github.com/AndPotap/afr/blob/main/utils/common_utils.py#L178
import json
import os
import random
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _bincount_array_as_tensor(arr):
    return torch.from_numpy(np.bincount(arr)).long()


def _randomly_split_in_two(samples, prop=0.8, seed=21, max_prop=1.0):
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[: int(len(samples) * max_prop)]
    first_total = int(len(samples) * np.abs(prop))
    if prop >= 0:
        return samples[:first_total]
    else:
        return samples[-first_total:]


def get_group_array(y_array, spurious_array, all_groups):
    group_array = np.zeros(y_array.shape[0], dtype=np.int64)
    for idx, (y, ss) in enumerate(all_groups):
        mask1 = np.array(spurious_array == ss)
        mask2 = np.array(y_array == y)
        mask = np.logical_and(mask1, mask2)
        group_array[mask] = idx
    return group_array


def _get_split(split):
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise (f"Unknown split {split}")


def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError


class SpuriousDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, prop=1.0, seed=21, max_prop=1.0):
        self.basedir = basedir
        self.transform = transform
        self.split = split

        self.metadata_df = self._get_metadata(split)
        indices = np.arange(len(self.metadata_df))
        ind = _randomly_split_in_two(indices, prop=prop, seed=seed, max_prop=max_prop)
        self.metadata_df = self.metadata_df.iloc[np.sort(ind)]

        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = _bincount_array_as_tensor(self.y_array)
        self.spurious_counts = _bincount_array_as_tensor(self.spurious_array)

    def _count_groups(self):
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.n_groups = len(self.group_counts)
        self.active_groups = np.unique(self.group_array).tolist()

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(self.y_array * self.n_spurious + self.spurious_array)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        domain = self.spurious_array[idx]
        group = self.group_array[idx]
        x = self._image_getitem(idx)
        file_path = self.filename_array[idx]
        return x, y, domain, group, file_path

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __str__(self):
        out_str = "\n".join(
            [
                f"Dataset {self.__class__.__name__} in {self.basedir}",
                f"Split: {self.split}",
                f"Number of samples: {len(self)}",
                f"Number of classes: {self.n_classes} ({self.y_counts.tolist()})",
                f"Number of spurious: {self.n_spurious} ({self.spurious_counts.tolist()})",
                f"Number of groups: {self.n_groups} ({self.group_counts.tolist()})",
                f"Transform: {self.transform}",
            ]
        )
        return out_str + "\n"


class JTTSpuriousDataset(SpuriousDataset):
    def __init__(
        self, basedir, subset, upweight, split="train", transform=None, prop=1.0, max_prop=1.0
    ):
        del prop
        del max_prop
        selected_cols = ["img_filename", "y", "split", "place"]
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)
        mistakes = pd.DataFrame({"img": subset * int(upweight - 1)})
        aux = pd.merge(
            left=self.metadata_df, right=mistakes, left_on="img_filename", right_on="img"
        )
        self.metadata_df = pd.concat((self.metadata_df[selected_cols], aux[selected_cols]))
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values


class ShrinkedSpuriousDataset(SpuriousDataset):
    def __init__(self, basedir, subset, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)
        subset = subset.reset_index()
        subset = subset.rename(columns={"index": "img", 0: "group", 1: "focal"})
        aux = pd.merge(left=self.metadata_df, right=subset, left_on="img_filename", right_on="img")
        self.metadata_df = aux[["img_filename", "y", "split", "place"]]
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values


class PhasesDataset(SpuriousDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)

    def __getitem__(self, idx):
        file_path = self.filename_array[idx]
        label = self.y_array[idx]
        group = self.group_array[idx]
        is_spurious = self.spurious_array[idx]
        image = self._image_getitem(idx)
        return file_path, image, label, group, is_spurious


class EmbeddingsDataset:
    def __init__(self, embeddings, targets, weights):
        self.embeddings = embeddings
        self.targets = targets
        self.weights = weights

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        y = self.targets[idx]
        w = self.weights[idx]
        return emb, y, w


def remove_minority_groups(trainset, num_remove):
    if num_remove == 0:
        return
    print("Removing minority groups")
    print("Initial groups", np.bincount(trainset.group_array))
    group_counts = trainset.group_counts
    minority_groups = np.argsort(group_counts.numpy())[:num_remove]
    minority_groups
    idx = np.where(
        np.logical_and.reduce([trainset.group_array != g for g in minority_groups], initial=True)
    )[0]
    trainset.y_array = trainset.y_array[idx]
    trainset.group_array = trainset.group_array[idx]
    trainset.confounder_array = trainset.confounder_array[idx]
    trainset.filename_array = trainset.filename_array[idx]
    trainset.metadata_df = trainset.metadata_df.iloc[idx]
    print("Final groups", np.bincount(trainset.group_array))


def balance_groups(ds):
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[:min_group] for idx in group_idx]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


def unbalance_groups(ds, group_ratios):
    # keep as much data as possible but with the given group ratios,
    # assuming original groups roughly balanced scale ratios so that largest one is 1
    group_ratios = np.array(group_ratios)
    group_ratios = group_ratios / np.max(group_ratios)
    print("Unbalancing groups with ratios", group_ratios)
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[: int(min_group * r)] for idx, r in zip(group_idx, group_ratios)]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


def subsample(ds, frac, generator=torch.Generator().manual_seed(42)):
    subsample_size = int(len(ds) * frac)
    idx = torch.randperm(len(ds), generator=generator).tolist()[:subsample_size]
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))


def subsample_to_size_and_ratio(ds, final_size, final_ratio):
    # subsample ds so that the final dataset has final_size samples and
    # final_ratio for each group where final_ratio sums to 1
    final_ratio = np.array(final_ratio)
    final_ratio = final_ratio / np.sum(final_ratio)
    print(f"Subsampling to size {final_size} samples with ratios {final_ratio}")
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    check = (final_size * final_ratio <= group_counts).all()
    assert check, "Cannot subsample to desired size and ratio"
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    group_idx = [idx[: int(final_size * r)] for idx, r in zip(group_idx, final_ratio)]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


def subset(ds, idx):
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))


def concate(ds1, ds2):
    ds1.y_array = np.concatenate((ds1.y_array, ds2.y_array), axis=0)
    ds1.group_array = np.concatenate((ds1.group_array, ds2.group_array), axis=0)
    ds1.spurious_array = np.concatenate((ds1.spurious_array, ds2.spurious_array), axis=0)
    ds1.filename_array = np.concatenate((ds1.filename_array, ds2.filename_array), axis=0)
    ds1.metadata_df = pd.concat((ds1.metadata_df, ds2.metadata_df), axis=0)
    ds1.group_counts = torch.from_numpy(np.bincount(ds1.group_array))


class DatasetGroup:
    def __init__(self, data):
        self.x, self.y = data

    def __getitem__(self, index):
        return self.x[index], int(self.y[index]), 0, 0

    def __len__(self):
        return self.x.shape[0]


def fill_spurious(x, y, categories, alpha, spurious_dim, locs=(1.0, 0.0), scales=(0.1, 10.0)):
    spur_mean, noise_mean = locs
    spur_std, noise_std = scales
    mask = np.isin(y, categories)

    spurious_loc = np.random.choice(a=2, size=(x.shape[0], 1), p=[alpha, 1 - alpha])
    x_new = np.random.normal(loc=noise_mean, scale=noise_std, size=(x.shape[0], spurious_dim))
    spur_feat = np.random.normal(loc=spur_mean, scale=spur_std, size=(x.shape[0], spurious_dim))
    x_new[mask] = np.where(spurious_loc[mask] == 0, spur_feat[mask], x_new[mask])

    x_new = np.concatenate((x, x_new), axis=1)
    return x_new


class WaterbirdsDataset(SpuriousDataset):
    def __init__(self, split, transform, basedir="data/waterbird_complete95_forest2water2"):
        super().__init__(
            basedir=basedir, split=split, transform=transform, prop=1.0, seed=42, max_prop=1.0
        )


class CelebADataset(SpuriousDataset):
    def __init__(self, split, transform, basedir="data/CelebA"):
        super().__init__(
            basedir=basedir, split=split, transform=transform, prop=1.0, seed=42, max_prop=1.0
        )


class DomainLabelDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        for d_idx, dataset in enumerate(self.datasets):
            if i < len(dataset):
                x, y = dataset[i]
                return x, y, d_idx
            i -= len(dataset)

    def __str__(self):
        data_info = "DomainLabelDataset(\n"
        data_info += f"  Number of domains: {len(self.datasets)}\n"
        for d_idx, dataset in enumerate(self.datasets):
            data_info += f"  Domain {d_idx}: {dataset.__str__()}\n"
        data_info += ")"
        return data_info

    def __len__(self):
        return sum(len(d) for d in self.datasets)


def file_cache(filename):
    """Decorator to cache the output of a function to disk."""

    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out

        return decorated

    return decorator


dataset_info = {
    "waterbirds": {
        "data_obj": WaterbirdsDataset,
        "data_root": "data/waterbird_complete95_forest2water2",
        "num_classes": 2,
        "idx_to_class": {0: "landbird", 1: "waterbird"},
        "class_to_idx": {"landbird": 0, "waterbird": 1},
        "group_split": np.array([3498, 184, 56, 1057]).astype(float),
        "group_names": ["LB on L", "LB on W", "WB on L", "WB on W"],
    },
    "celeba": {
        "data_obj": CelebADataset,
        "data_root": "data/CelebA",
        "num_classes": 2,
        "idx_to_class": {0: "nonblond", 1: "blond"},
        "class_to_idx": {"nonblond": 0, "blond": 1},
        "group_split": np.array([71629, 66874, 22880, 1387]).astype(float),
        "group_names": ["NB F", "NB M", "B F", "B M"],
    },
}

if __name__ == "__main__":
    from torchvision import transforms

    totensor = transforms.ToTensor()
    for name, ddict in dataset_info.items():
        data_obj = ddict["data_obj"]
        train_data = data_obj(split="train", transform=totensor)
        val_data = data_obj(split="val", transform=totensor)
        test_data = data_obj(split="test", transform=totensor)

        print(name)
        print(train_data)
        print(val_data)
        print(test_data)
        print(test_data[0][0].shape)
        print(val_data.filename_array[0], "\n\n")
