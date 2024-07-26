# This file implements most of the training and data logic.
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm

from backbones import feedforward_text, get_outputs
from datasets_all import dataset_info

logger.remove()
log_format = "<green>{time:HH:mm:ss}</green> <level>{message}</level>"
logger.add(sys.stdout, colorize=True, format=log_format)
logfile_name = f"logs/runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger.add(logfile_name, format=log_format)

torch.set_num_threads(1)

# Change backbone_name to the backbone you want to use; we use ResNet-50 by default
# backbone_name = "B32-best"
backbone_name = "R50-openai"
# backbone_name = "L14-openai"
# backbone_name = "L14-best"


@lru_cache(maxsize=None)
def get_cached_data(dataset_name, splits=None):
    """Return a dictionary of features, labels, and domains for a dataset. Handles normalization and whitening of features."""
    print(f"Getting cached data for {dataset_name}...")
    normalize = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)

    mean_and_std_fn = f"features/{dataset_name}-{backbone_name}-mean_and_std.pkl"
    if os.path.exists(mean_and_std_fn):
        """Loading cached mean and std from train split"""
        with open(mean_and_std_fn, "rb") as f:
            tr_f_mean, tr_f_std = pickle.load(f)
    else:
        tr_outputs = get_outputs(dataset_name, backbone_name, "train")
        tr_x = tr_outputs["features"]
        tr_f_n = normalize(tr_x)
        tr_f_mean, tr_f_std = tr_f_n.mean(0), tr_f_n.std(0)
        with open(mean_and_std_fn, "wb") as f:
            pickle.dump((tr_f_mean, tr_f_std), f)

    if splits is None:
        splits = ["train", "val", "test"]
    all_outputs = dict()
    for split in splits:
        all_outputs[split] = get_outputs(dataset_name, backbone_name, split)

    data = dict()
    for split in all_outputs.keys():
        split_features = all_outputs[split]["features"]
        split_labels = all_outputs[split]["labels"]
        split_domains = all_outputs[split]["domains"]
        norm_split_features = normalize(split_features)
        data[split] = {
            "labels": torch.from_numpy(split_labels).long(),
            "domains": torch.from_numpy(split_domains).long(),
            "feats": torch.from_numpy(split_features).float(),
            "feats_n": torch.from_numpy(norm_split_features).float(),
        }
        whiten = lambda x: (x - tr_f_mean) / tr_f_std
        data[split]["feats_n_w"] = whiten(data[split]["feats_n"])
    return data


@torch.no_grad()
def evaluate(net, x, y, masks=None):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    logits = net(x).detach().cpu()
    y = y.detach().cpu()
    corrects = logits.argmax(1) == y
    loss = criterion(logits, y).float().mean().item()
    if masks is None:
        l_masks = [(y == l).bool() for l in np.unique(y)]
        l_losses = torch.stack([criterion(logits[m], y[m]).mean() for m in l_masks])
        crossval_loss = l_losses.max().item()
        acc = corrects.float().mean().item()
    else:
        raw_loss = criterion(logits, y)
        ml_loss = torch.stack([raw_loss[m].mean() for m in masks])
        ml_acc = torch.stack([corrects[m].float().mean() for m in masks])
        ml_loss = ml_loss[~torch.isnan(ml_loss)]
        ml_acc = ml_acc[~torch.isnan(ml_acc)]

        crossval_loss = ml_loss.max().item()
        acc = ml_acc.min().item()
    return {
        "crossval_loss": crossval_loss,  # cross-validation with worst-group loss
        "loss": loss,
        "acc": acc,
        "corrects": corrects.numpy(),
        "logits": logits.numpy(),
    }


def get_loss(preds, labels, masks):
    lbl_cpu = labels.cpu()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    raw_loss = criterion(preds, labels)
    if masks is None:  # Standard ERM, no group splits
        split_masks = [(lbl_cpu == l).bool() for l in np.unique(lbl_cpu)]
        split_losses = torch.stack([raw_loss[m].mean() for m in split_masks])

        loss = raw_loss.mean()  # standard cross-entropy
        # loss = split_losses.mean()  # class-balanced ERM
        # loss = split_losses.max()  # worst-class ERM
    else:
        split_losses = torch.stack([raw_loss[m].mean() for m in masks])
        split_losses = split_losses[~torch.isnan(split_losses)]  # filter out NaNs

        # loss = split_losses.mean() # DFR
        loss = split_losses.max()  # Group DRO
    return loss


def _train(steps, all_data, num_classes, masks, lr, weight_decay):
    """One run of training."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if masks is not None:
        masks = [m.to(dtype=torch.bool) for m in masks]
    tr_N, D = all_data["train"]["feats_n_w"].shape
    net = torch.nn.Linear(D, num_classes)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    tr_idxs, val_idxs = train_test_split(np.arange(tr_N), test_size=0.2)
    data = {
        "x_tr": all_data["train"]["feats_n_w"][tr_idxs],
        "y_tr": all_data["train"]["labels"][tr_idxs],
        "x_id_val": all_data["train"]["feats_n_w"][val_idxs],
        "y_id_val": all_data["train"]["labels"][val_idxs],
        "x_val": all_data["val"]["feats_n_w"],
        "y_val": all_data["val"]["labels"],
        "x_test": all_data["test"]["feats_n_w"],
        "y_test": all_data["test"]["labels"],
    }
    data = {k: v.to(device) for k, v in data.items()}
    if masks is not None:
        data["m_tr"] = [m[tr_idxs] for m in masks]
        data["m_id_val"] = [m[val_idxs] for m in masks]
    else:
        data["m_tr"] = None
        data["m_id_val"] = None
    net.to(device)

    patience = 0
    best_val_loss = float("inf")
    for i in tqdm(range(steps)):
        preds = net(data["x_tr"])
        loss = get_loss(preds, data["y_tr"], data["m_tr"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss = get_loss(net(data["x_id_val"]), data["y_id_val"], data["m_id_val"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            crossval_metrics = {
                "id_val": evaluate(net, data["x_id_val"], data["y_id_val"], masks=data["m_id_val"]),
                # "val": evaluate(net, data["x_val"], data["y_val"]),
                # "test": evaluate(net, data["x_test"], data["y_test"]),
            }
        else:
            patience += 1
            if patience > 20:
                break
    crossval_metrics["net"] = net.cpu()
    return crossval_metrics


def calculate_accs(logits, labels, domains):
    corrects = logits.argmax(1) == np.array(labels)
    metrics = {"avg": corrects.mean().item()}
    num_d = len(np.unique(domains))
    groups = np.array(labels) * num_d + np.array(domains)
    groups_sorted = sorted(np.unique(groups))
    groupwise_accs = {g: corrects[groups == g].mean() for g in groups_sorted}
    metrics.update(groupwise_accs)
    return metrics


def get_classwise_metrics(nets, data):
    assert isinstance(nets, list)
    x, y = data["feats_n_w"], data["labels"]
    evals = [evaluate(net, x, y) for net in nets]
    logits = np.stack([e["logits"] for e in evals])
    metrics = {}
    corrects = logits.argmax(-1) == np.array(y)
    metrics["probs"] = torch.tensor(logits).softmax(-1).mean(0).numpy()
    metrics["preds"] = metrics["probs"].argmax(-1)
    metrics["acc"] = corrects.mean().item()
    metrics["classwise_accs"] = [
        corrects[:, y == class_idx].mean().item() for class_idx in np.unique(y)
    ]
    metrics["labels"] = y
    print(f"Overall acc: {metrics['acc']*100:.2f}%")
    return metrics


def get_metrics(nets, data):
    """Takes list of nets, returns metrics for each net."""
    assert isinstance(nets, list)
    x, y, d = data["feats_n_w"], data["labels"], data["domains"]
    evals = [evaluate(net, x, y) for net in nets]
    _metrics = [calculate_accs(e["logits"], y, d) for e in evals]
    metrics = {k: np.stack([m[k] for m in _metrics]) for k in _metrics[0].keys()}
    logits = np.stack([e["logits"] for e in evals])
    probs = torch.tensor(logits).softmax(-1).mean(0).numpy()
    preds = probs.argmax(1)
    metrics.update({"probs": probs, "preds": preds})
    return metrics


def get_training_results(dataset_name, subset_idx=None, masks=None, max_steps=3000, repeats=3):
    """Full training loop with hyperparameter search."""
    all_data = get_cached_data(dataset_name)
    if subset_idx is not None:
        # If subset_idx is specified, only use a subset of the training data
        assert all(idx in range(all_data["train"]["feats_n_w"].shape[0]) for idx in subset_idx)
        all_data["train"] = {k: v[subset_idx] for k, v in all_data["train"].items()}
        if masks is not None:
            masks = [m[subset_idx] for m in masks]

    num_classes = dataset_info[dataset_name]["num_classes"]
    if dataset_name == "waterbirds":
        lrs = [0.01]
        wds = [0.01]
    elif dataset_name == "celeba":
        lrs = [0.1]
        wds = [0.01]

    logger.info(f"Dataset={dataset_name}. Hyperparameter grid lrs={lrs}, wds={wds}")
    param_grid = {"lr": lrs, "weight_decay": wds}
    grid = ParameterGrid(param_grid)
    best_loss = float("inf")
    for hparams in grid:
        logger.info(f"Training with hyperparameters: {hparams}")
        metrics_all = []
        for _ in range(repeats):
            _metrics = _train(
                max_steps, all_data, num_classes, masks, hparams["lr"], hparams["weight_decay"]
            )
            metrics_all.append(_metrics)
        id_val_losses = [m["id_val"]["crossval_loss"] for m in metrics_all]
        avg_loss = np.mean(id_val_losses)

        if avg_loss < best_loss:
            best_hparams, best_loss = hparams, avg_loss
            best_nets = [m["net"] for m in metrics_all]

    logger.info(f"Best params = {best_hparams}")
    full_results = dict()
    for split in ["val", "test"]:
        split_metrics = get_metrics(best_nets, all_data[split])
        full_results[f"{split}_avg"] = split_metrics["avg"].mean()

        if not "group_split" in dataset_info[dataset_name]:
            logger.info(f"full_results[{split}_avg] = {full_results[f'{split}_avg']}")
            text_items = [f"Avg: {full_results[f'{split}_avg']*100:.1f}"]
            full_results[f"{split}_text"] = "\n".join(text_items)
            continue

        groups = np.array([0, 1, 2, 3])
        split_group_grid = np.stack([split_metrics[g] for g in groups])
        full_results[f"{split}_worst"] = np.min(split_group_grid, axis=0).mean()

        group_split = dataset_info[dataset_name]["group_split"]
        train_group_ratio = group_split / group_split.sum()
        tr_w_metrics = split_group_grid * train_group_ratio[:, None]
        full_results[f"{split}_adj_avg"] = np.sum(tr_w_metrics, axis=0).mean()
        full_results[f"{split}_gap"] = (
            full_results[f"{split}_adj_avg"] - full_results[f"{split}_worst"]
        )

        logger.info(
            f"{full_results[f'{split}_worst']*100:.1f} & {full_results[f'{split}_adj_avg']*100:.1f} & {full_results[f'{split}_gap']*100:.1f} & "
        )
        text_items = [f"Adj Avg:  {full_results[f'{split}_adj_avg']*100:.1f}"]
        text_items.append(
            f"Worst:    {full_results[f'{split}_worst']*100:.1f} (gap={full_results[f'{split}_gap']*100:.1f})"
        )
        split_group_avg = split_group_grid.mean(axis=1)
        group_names = dataset_info[dataset_name]["group_names"]
        group_text = "  ".join(
            [f"{gname}={split_group_avg[g]*100:.1f}" for g, gname in zip(groups, group_names)]
        )
        text_items.append(group_text)
        full_results[f"{split}_text"] = "\n".join(text_items)

    return {
        "nets": best_nets,
        "metrics": full_results,
        "val_text": full_results["val_text"],
        "test_text": full_results["test_text"],
    }


def train_with_groups(dataset_name):
    all_data = get_cached_data(dataset_name)
    train_y = all_data["train"]["labels"]
    train_d = all_data["train"]["domains"]
    num_d = len(np.unique(train_d))
    train_g = train_y * num_d + train_d
    group_masks = [train_g == g for g in np.unique(train_g)]
    return get_training_results(dataset_name=dataset_name, masks=group_masks)


def get_text_embedding(_prompts):
    prompts = [f"an image of a {p}" for p in _prompts]

    print(f"Getting text embedding for {len(prompts)} prompts...")
    cache_path = f"features/textemb_{backbone_name}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if isinstance(prompts, str):
        prompt = prompts
        if prompt in cache:
            return cache[prompt]
        else:
            text_features = feedforward_text([prompt], backbone_name)
            cache[prompt] = text_features[0]
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            return cache[prompt]
    else:
        noncached_prompts = list(filter(lambda x: x not in cache, prompts))
        if len(noncached_prompts) > 0:
            text_features = feedforward_text(noncached_prompts, backbone_name)
            for prompt, text_feature in zip(noncached_prompts, text_features):
                cache[prompt] = text_feature
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

        return np.stack([cache[p] for p in prompts])


def get_similarities(prompt, split, dataset_name):
    all_data = get_cached_data(dataset_name)
    n_img_embs = np.array(all_data[split]["feats_n"])
    text_embedding = get_text_embedding(prompt)
    n_text_embs = text_embedding / np.linalg.norm(text_embedding)
    similarities = n_text_embs @ n_img_embs.T
    return similarities


def get_masks(reweight_dict):
    dataset_name = reweight_dict["dataset_name"]
    all_masks = []
    for class_name, params in reweight_dict["cutoffs"].items():
        prompt, cutoff = params["prompt"], params["sim_cutoff"]
        class_to_idx = dataset_info[dataset_name]["class_to_idx"]
        class_idx = class_to_idx[class_name]
        all_data = get_cached_data(dataset_name)
        class_mask = all_data["train"]["labels"] == class_idx

        train_similarities = get_similarities(prompt, "train", dataset_name)
        attribute_mask = train_similarities > cutoff

        left_mask = np.logical_and(class_mask, ~attribute_mask)
        right_mask = np.logical_and(class_mask, attribute_mask)
        all_masks += [left_mask, right_mask]

        n_left, n_right = left_mask.sum(), right_mask.sum()
        frac_left, frac_right = n_left / class_mask.sum(), n_right / class_mask.sum()
        logger.info(f"Class {class_name} has {class_mask.sum()} images total")
        logger.info(f"{n_left} / {n_right} split ({frac_left*100:.1f}% / {frac_right*100:.1f}%)")
    return all_masks


def reweight_and_train(reweight_dict):
    dataset_name = reweight_dict["dataset_name"]
    all_masks = get_masks(reweight_dict)
    training_results = get_training_results(dataset_name=dataset_name, masks=all_masks)

    fn = f"leaderboard_{dataset_name}_{backbone_name}.json"
    leaderboard_path = os.path.join("logs", fn)
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, "r") as f:
            current_leaderboard = json.load(f)
    else:
        current_leaderboard = []
    metrics_to_save = [
        "val_avg",
        "val_adj_avg",
        "val_worst",
        "val_gap",
        "test_avg",
        "test_adj_avg",
        "test_worst",
        "test_gap",
    ]
    metrics_dict = {k: training_results["metrics"][k] for k in metrics_to_save}
    metrics_dict["cutoff_params"] = reweight_dict["cutoffs"]
    current_leaderboard.append(metrics_dict)
    with open(leaderboard_path, "w") as f:
        json.dump(current_leaderboard, f, indent=4)

    return training_results


def get_nets(dataset_name, zero_shot=False):
    """Get cached linear probe networks for a dataset."""
    os.makedirs("checkpoints", exist_ok=True)
    fn = f"checkpoints/{backbone_name}-{dataset_name}.pkl"
    if os.path.exists(fn):
        print(f"Loading cached networks for {dataset_name}...")
        with open(fn, "rb") as f:
            nets = pickle.load(f)
        return nets

    print(f"Training networks for {dataset_name}...")
    training_results = get_training_results(dataset_name, repeats=1)
    nets = training_results["nets"]
    with open(fn, "wb") as f:
        pickle.dump(nets, f)
    return nets


def small_data_experiment(dataset_name):
    all_results = defaultdict(list)

    if dataset_name == "waterbirds":
        repeats = 25
        run_repeats = 5
        Ns = [75, 100, 200, 300, 400, 500, 1000, 2000]
        prompt = "a bird in the ocean"
        c1 = c2 = 0.18
        wbird_setting = {
            "dataset_name": dataset_name,
            "cutoffs": {
                "landbird": {"prompt": prompt, "sim_cutoff": c1},
                "waterbird": {"prompt": prompt, "sim_cutoff": c2},
            },
        }
    elif dataset_name == "celeba":
        Ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
        repeats = 5
        run_repeats = 5
        prompt = "an image of a man"
        c1 = c2 = 0.17
        wbird_setting = {
            "dataset_name": dataset_name,
            "cutoffs": {
                "nonblond": {"prompt": prompt, "sim_cutoff": c1},
                "blond": {"prompt": prompt, "sim_cutoff": c2},
            },
        }

    all_data = get_cached_data(dataset_name=dataset_name)
    train_N = all_data["train"]["labels"].shape[0]
    for N in Ns:
        for _ in range(repeats):
            subset_idx = np.random.choice(train_N, N, replace=False)
            _results = get_training_results(
                dataset_name=dataset_name, subset_idx=subset_idx, repeats=run_repeats
            )
            all_results[f"random_{N}_avg"] = _results["metrics"]["test_adj_avg"]
            all_results[f"random_{N}_worst"] = _results["metrics"]["test_worst"]

    masks = get_masks(wbird_setting)
    num_masks = len(masks)
    for N in Ns:
        for _ in range(repeats):
            imgs_per_mask = N // num_masks
            subset_idx = []
            for m in masks:
                idx = np.where(m)[0]
                mask_N = min(imgs_per_mask, len(idx))
                subset_idx += list(np.random.choice(idx, mask_N, replace=False))
            if len(subset_idx) < N:  # add more images not yet sampled
                not_sampled = np.array(list(set(range(train_N)) - set(subset_idx)))
                subset_idx += list(np.random.choice(not_sampled, N - len(subset_idx)))
            assert len(subset_idx) == N

            _results = get_training_results(
                dataset_name=dataset_name, subset_idx=subset_idx, masks=masks, repeats=run_repeats
            )
            all_results[f"interact_{N}_avg"] = _results["metrics"]["test_adj_avg"]
            all_results[f"interact_{N}_worst"] = _results["metrics"]["test_worst"]

    avg_results = {k: np.mean(v) for k, v in all_results.items()}
    aggregated_results = {
        "random_avg": [avg_results[f"random_{N}_avg"] for N in Ns],
        "random_worst": [avg_results[f"random_{N}_worst"] for N in Ns],
        "interact_avg": [avg_results[f"interact_{N}_avg"] for N in Ns],
        "interact_worst": [avg_results[f"interact_{N}_worst"] for N in Ns],
    }

    logger.info(Ns)
    logger.info(", ".join(map(lambda x: f"{x*100:.1f}", aggregated_results["interact_avg"])))
    logger.info(", ".join(map(lambda x: f"{x*100:.1f}", aggregated_results["interact_worst"])))
    logger.info(", ".join(map(lambda x: f"{x*100:.1f}", aggregated_results["random_avg"])))
    logger.info(", ".join(map(lambda x: f"{x*100:.1f}", aggregated_results["random_worst"])))
