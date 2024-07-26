import itertools
import json
import logging
import os
import pickle
import re
import shutil
import time
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from blip_caption import extract_keyword, get_captions
from datasets_all import dataset_info
from main import get_cached_data, get_classwise_metrics, get_nets, get_text_embedding

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)  # Generates a new secret key on each run
data = defaultdict(list)

log_fn = os.path.join("logs", "logs.txt")
os.makedirs("logs", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)
os.makedirs(f"session_data", exist_ok=True)
for subdir in os.listdir("session_data"):
    subpath = os.path.join("session_data", subdir)
    if os.path.isdir(subpath):
        shutil.rmtree(subpath)

# Google CDN for fast image loading.
cdn_root = f"https://storage.googleapis.com/interactive-static-2"
gcs_url_for = lambda filename: os.path.join(cdn_root, filename)
app.jinja_env.globals.update(gcs_url_for=gcs_url_for)
default_dataset_name = "dsprites"
dataset_sequence = ["dsprites", "waterbirds", "celeba"]
IMGS_IN_GRID = 70


def load_dataset(dataset_name):
    print(f"Loading dataset {dataset_name}")

    if dataset_name == "dsprites":
        data = get_data()
        full_fns = data["metrics"]["filenames"]
        return None, full_fns
    elif dataset_name in ["waterbirds", "celeba"]:
        dataset_obj = dataset_info[dataset_name]["data_obj"]
        all_cached_data = get_cached_data(dataset_name, splits=("val",))
        cached_data = all_cached_data["val"]
        raw_val_dataset = dataset_obj(split="val", transform=None)
        fns = list(raw_val_dataset.filename_array)
        full_fns = [f"{dataset_name}_small/{fn}" for fn in fns]
    del raw_val_dataset
    return cached_data, full_fns


@app.route("/change_dataset", methods=["POST"])
def change_dataset():
    """Change the dataset and reinitialize the session data."""
    dataset_name = request.form.get("dataset_name")
    session["dataset_name"] = dataset_name

    # Reset the session data
    session_id = session["session_id"]
    shutil.rmtree(f"session_data/{session_id}", ignore_errors=True)
    return redirect(url_for("home"))


def get_data():
    session_id = session["session_id"]
    fn = os.path.join("session_data", session_id, "data.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            data = pickle.load(f)
    else:
        data = None
    return data


def save_data(data):
    def numpy_to_list(data):
        # Convert nested data into json-serializable format
        if isinstance(data, dict):
            return {k: numpy_to_list(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data.astype(float).tolist()
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(float).tolist()
        elif isinstance(data, np.float32):
            return float(data)
        elif isinstance(data, list):
            return [numpy_to_list(v) for v in data]
        else:
            return data

    session_id = session["session_id"]
    fn = os.path.join("session_data", session_id, "data.pkl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    data = numpy_to_list(data)
    with open(fn, "wb") as f:
        pickle.dump(data, f)


@app.route("/select_class", methods=["POST"])
def select_class():
    """Save selected class to session data."""
    data = get_data()
    class_name = request.form["class_name"]
    data["selected_class"] = class_name
    data.pop("prompt_suggestions")
    save_data(data)
    sample_images()
    if "slider_value" in data:
        split_similarities(data["slider_value"])
    return redirect(url_for("home"))


@app.route("/process_button", methods=["POST"])
def process_button():
    """Process prompt button press."""
    prompt = request.form["prompt"]
    print(f"Processing prompt: {prompt}")

    data = get_data()
    if "slider_value" in data:
        del data["slider_value"]
    data["prompt"] = prompt
    save_data(data)
    sample_images()
    return "Success", 200


def _process_text(user_text):
    data = get_data()
    data["prompt"] = user_text
    if "all_prompts" not in data:
        data["all_prompts"] = []
    else:
        if user_text in data["all_prompts"]:
            data["all_prompts"].remove(user_text)
    data["all_prompts"].append(user_text)
    data["all_prompts"] = data["all_prompts"][-10:]  # limit history to 10

    if "slider_value" in data:
        del data["slider_value"]
    save_data(data)
    sample_images()

    with open(log_fn, "a") as f:
        time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        sid = session["session_id"][:8]
        f.write(f"{sid}\t{time}\t{user_text}\n")

    data = get_data()
    return jsonify(data)


@app.route("/process_text", methods=["POST"])
def process_text():
    request_data = request.get_json()
    user_text = request_data["user_text"]

    prolific_id = request_data["prolificID"]
    dataset = session.get("dataset_name", default=default_dataset_name)
    log_dir = "logs/prolific_text_logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/{prolific_id}.txt", "a") as f:
        sid = session["session_id"]
        time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{dataset}\t{user_text}\n")
    return _process_text(user_text)


@app.route("/reset", methods=["POST"])
def retrain():
    """Reset button"""
    print("Resetting...")
    session["prompts"] = []
    session_id = session["session_id"]
    shutil.rmtree(f"session_data/{session_id}", ignore_errors=True)
    return redirect(url_for("home"))


def get_error_score(similarities):
    """Calculate the class-balanced binary classification accuracy of the best threshold classifier based on error score."""
    assert len(similarities) == 2
    similarities = [np.array(sim) for sim in similarities]
    all_sims = np.concatenate(similarities)
    min_sim, max_sim = np.min(all_sims), np.max(all_sims)
    thresholds = np.linspace(min_sim, max_sim, 1000)
    best_acc = 0
    best_threshold = 0
    for threshold in thresholds:
        preds = [similarities[0] < threshold, similarities[1] >= threshold]
        accs_per_class = [np.mean(pred) for pred in preds]
        acc = np.mean(accs_per_class)
        acc = max(acc, 1 - acc)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    normalized_best_acc = (best_acc - 0.5) * 2
    return normalized_best_acc, best_threshold


def get_prompt_suggestions(dataset_name, model_predictions, selected_idx):
    fn = f"features/prompt_suggestions_{dataset_name}.pkl"
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            all_suggestions = pickle.load(f)
        if selected_idx in all_suggestions:
            return all_suggestions[selected_idx]
    else:
        all_suggestions = dict()

    outputs, full_fns = load_dataset(dataset_name)
    pred_is_selected = model_predictions == selected_idx
    class_is_selected = np.array(outputs["labels"]) == selected_idx
    pred_class = np.where(pred_is_selected & class_is_selected)[0]
    nopred_class = np.where(~pred_is_selected & class_is_selected)[0]

    captions = get_captions(dataset_name=dataset_name, split="val")
    pred_class_caps = [captions[idx] for idx in pred_class]
    nopred_class_caps = [captions[idx] for idx in nopred_class]
    pred_class_keywords = extract_keyword(pred_class_caps)
    nopred_class_keywords = extract_keyword(nopred_class_caps)
    all_keywords = pred_class_keywords + nopred_class_keywords

    remove_words = ["the", "is", "of", "are"]
    for word in remove_words:
        all_keywords = [keyword.replace(f" {word} ", " ") for keyword in all_keywords]
    all_keywords = [keyword.strip() for keyword in all_keywords]
    subphrases = []
    for phrase in all_keywords:
        words = re.findall(r"\b\w+\b", phrase)
        subphrases += words
    all_keywords += subphrases
    all_keywords = list(set(all_keywords))

    text_embs = get_text_embedding(all_keywords)
    n_text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
    n_img_embs = outputs["feats_n"]
    all_sims = np.dot(n_text_embs, n_img_embs.T)

    candidate_results = dict()
    for keyword, sims in zip(all_keywords, all_sims):
        score, threshold = get_error_score([sims[pred_class], sims[nopred_class]])
        print(f"{keyword:20s}: {score:.2f}")
        candidate_results[keyword] = score
    candidate_results = {
        k: v for k, v in sorted(candidate_results.items(), key=lambda item: item[1], reverse=True)
    }

    all_suggestions[selected_idx] = candidate_results
    with open(fn, "wb") as f:
        pickle.dump(all_suggestions, f)
    return candidate_results


def sample_images():
    data = get_data()
    metrics = data["metrics"]
    dataset_name = session.get("dataset_name", default=default_dataset_name)
    outputs, full_fns = load_dataset(dataset_name)

    selected_class = data["selected_class"]
    if dataset_name == "dsprites":
        selected_idx = 1
        y = metrics["labels"]
        s = metrics["s_labels"]
        color_embs = get_text_embedding(["red", "blue"])
        shape_embs = get_text_embedding(["square", "ellipse"])
        color_embs_n = color_embs / np.linalg.norm(color_embs, axis=1, keepdims=True)
        shape_embs_n = shape_embs / np.linalg.norm(shape_embs, axis=1, keepdims=True)
        all_color_embs = color_embs_n[s]
        all_shape_embs = shape_embs_n[y]
        feats = all_color_embs + all_shape_embs
        feats += np.random.normal(scale=0.08, size=feats.shape)
        feats_n = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        outputs = {"labels": np.array(y), "feats_n": feats_n}
    else:
        selected_idx = dataset_info[dataset_name]["class_to_idx"][selected_class]

    data["selected_results"] = []
    preds = np.array(metrics["preds"]).astype(int)
    classes = np.array(outputs["labels"]).astype(int)
    pred_is_selected = preds == selected_idx
    class_is_selected = classes == selected_idx
    pred = np.where(pred_is_selected)[0]
    pred_class = np.where(pred_is_selected & class_is_selected)[0]
    nopred_class = np.where(~pred_is_selected & class_is_selected)[0]

    if "prompt_suggestions" not in data and dataset_name != "dsprites":
        candidate_results = get_prompt_suggestions(dataset_name, metrics["preds"], selected_idx)
        data["prompt_suggestions"] = [
            f"{k}: {v:.2f}" for k, v in list(candidate_results.items())[:10]
        ]

    def get_split_data(match_idxs, n=IMGS_IN_GRID):
        split_dict = dict()
        split_dict["match_idxs"] = match_idxs
        if len(match_idxs) < n:
            split_dict["selected_idxs"] = match_idxs
        else:
            split_dict["selected_idxs"] = np.random.choice(match_idxs, size=n, replace=False)
        split_dict["images"] = np.array(full_fns)[split_dict["selected_idxs"]].tolist()
        return split_dict

    pred_class_dict = get_split_data(pred_class)
    pred_noclass = np.setdiff1d(pred, pred_class)
    pred_noclass_dict = get_split_data(pred_noclass)
    nopred_class_dict = get_split_data(nopred_class)
    dataset_index = data["dataset_index"]
    if dataset_index == 0:
        pred_class_dict["text"] = f"Images of squares that the model predicted as squares"
        nopred_class_dict["text"] = f"Images of squares that the model predicted as ellipses"
    elif dataset_index == 1:
        pred_class_dict["text"] = f"Images of waterbirds that the model predicted as waterbirds"
        nopred_class_dict["text"] = f"Images of waterbirds that the model predicted as landbirds"
    elif dataset_index == 2:
        pred_class_dict["text"] = f"Images of blond people that the model predicted as blond"
        nopred_class_dict["text"] = f"Images of blond people that the model predicted as non-blond"
    pred_noclass_dict["text"] = f"Wrong Class (N={len(pred_noclass)})"
    pred_class_dict["correct_text"] = "correct"
    pred_noclass_dict["correct_text"] = "wrong_class"
    nopred_class_dict["correct_text"] = "wrong"
    pred_class_dict["correct"] = [True] * len(pred_class_dict["images"])
    nopred_class_dict["correct"] = [False] * len(nopred_class_dict["images"])

    prompt_data = dict()
    prompt_data["selected_results"] = [pred_class_dict, nopred_class_dict]
    # prompt_data["labels"] = outputs["labels"].tolist()
    prompt_data["correct"] = np.array(outputs["labels"]) == np.array(metrics["preds"])

    if not "prompt" in data:
        for split_dict in prompt_data["selected_results"]:
            split_dict["similarity"] = [0.0] * len(split_dict["images"])
        data["default_data"] = prompt_data
        save_data(data)
    else:
        text_emb = get_text_embedding([data["prompt"]])
        n_text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        n_all_img_emb = outputs["feats_n"]
        all_sims = np.dot(n_text_emb, n_all_img_emb.T).squeeze()
        min_sim, max_sim = all_sims.min(), all_sims.max()
        prompt_data["all_sims"] = all_sims.tolist()

        prompt_score, threshold = get_error_score([all_sims[pred_class], all_sims[nopred_class]])
        prompt_data["prompt_score"] = f"{prompt_score:.2f}"
        prompt_data["slider_value"] = threshold

        for split_dict in prompt_data["selected_results"]:
            selected_idxs = split_dict["selected_idxs"]
            raw_sims = all_sims[selected_idxs]
            normalized_sims = -1 + 2 * (raw_sims - min_sim) / (max_sim - min_sim)
            split_dict["similarity"] = normalized_sims.tolist()

        full_df = pd.DataFrame(columns=["text", "Similarity"])
        for split_dict in prompt_data["selected_results"]:
            temp_df = pd.DataFrame()
            N = len(split_dict["match_idxs"])
            temp_df["text"] = [split_dict["correct_text"]] * N
            temp_df["filename"] = np.array(full_fns)[split_dict["match_idxs"]]
            temp_df["Similarity"] = all_sims[split_dict["match_idxs"]]
            probs = np.array(metrics["probs"])[:, selected_idx]
            temp_df["confidence"] = probs[split_dict["match_idxs"]]
            full_df = pd.concat([full_df, temp_df])
        data_list = full_df.to_dict("records")
        json_fn = f"static/plots/{data['prompt']}.json"
        with open(json_fn, "w") as f:
            json.dump(data_list, f)
        prompt_data.update({"plot_json_fn": json_fn, "min_sim": min_sim, "max_sim": max_sim})
        session["entered_prompt"] = True

        current_prompt = data["prompt"]
        if "prompt_data" not in data:
            data["prompt_data"] = dict()
        data["prompt_data"][current_prompt] = prompt_data
        save_data(data)
        split_similarities()
    return "Success", 200


@app.route("/reweight", methods=["POST"])
def reweight():
    print(f"Submitting and moving on to the next task...")
    data = get_data()

    dataset = session.get("dataset_name", default=default_dataset_name)
    prompt = data["prompt"]
    cutoff = data["slider_value"]
    score = data["prompt_score"]
    task_str = f"{dataset}\t{prompt}\t{cutoff}\t{score}"

    with open(log_fn, "a") as f:
        time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        sid = session["session_id"][:8]
        f.write(f"{sid}\t{time}\t{task_str}\n")

    if dataset == "celeba":
        session["END"] = True
    current_idx = dataset_sequence.index(dataset)
    new_dataset = dataset_sequence[(current_idx + 1) % len(dataset_sequence)]
    session["dataset_name"] = new_dataset

    session["prompts"] = []
    session_id = session["session_id"]
    shutil.rmtree(f"session_data/{session_id}", ignore_errors=True)
    return "Success", 200


def cleanup_old_sessions(max_age=3600):
    now = time.time()
    too_old = lambda fn: now - os.path.getmtime(fn) > max_age

    for subdir in os.listdir("session_data"):
        subpath = os.path.join("session_data", subdir)
        if os.path.isdir(subpath) and too_old(subpath):
            shutil.rmtree(subpath)

    for fn in os.listdir("static/plots"):
        full_fn = os.path.join("static/plots", fn)
        if fn.endswith(".json") and too_old(full_fn):
            os.remove(full_fn)


def split_similarities():
    """Re-split similarities based on slider value."""
    data = get_data()
    current_prompt = data["prompt"]
    prompt_data = data["prompt_data"][current_prompt]
    all_sims = np.array(prompt_data["all_sims"])
    slider_value = prompt_data["slider_value"]
    labels = np.array(data["metrics"]["labels"])
    correct = np.array(data["metrics"]["preds"]) == labels

    dataset_name = session.get("dataset_name", default=default_dataset_name)
    _, full_fns = load_dataset(dataset_name)

    current_y = data["selected_class"]
    if dataset_name == "dsprites":
        current_y_idx = 1
    else:
        current_y_idx = dataset_info[dataset_name]["class_to_idx"][current_y]
    y_idxs = np.where(labels == current_y_idx)[0]
    y_sims = all_sims[y_idxs]

    more_similar_idx = y_idxs[np.where(y_sims > slider_value)[0]]
    more_N = len(more_similar_idx)
    if more_N >= IMGS_IN_GRID:
        more_sampled = np.random.choice(more_similar_idx, size=IMGS_IN_GRID, replace=False)
    else:
        more_sampled = more_similar_idx
    more_acc = correct[more_similar_idx].mean() * 100
    more_dict = {
        "text": f"Images with {current_prompt} ({int(more_acc)}% correct)",
        # "text": f"More similar (N={more_N}, {int(more_acc)}% correct)",
        "match_idxs": more_similar_idx.tolist(),
        "selected_idxs": more_sampled.tolist(),
        "images": np.array(full_fns)[more_sampled].tolist(),
        "similarity": all_sims[more_sampled].tolist(),
        "correct": correct[more_sampled].tolist(),
    }

    less_similar_idx = y_idxs[np.where(y_sims <= slider_value)[0]]
    less_N = len(less_similar_idx)
    if less_N >= IMGS_IN_GRID:
        less_sampled = np.random.choice(less_similar_idx, size=IMGS_IN_GRID, replace=False)
    else:
        less_sampled = less_similar_idx
    less_acc = correct[less_similar_idx].mean() * 100
    less_dict = {
        "text": f"Images without {current_prompt} ({int(less_acc)}% correct)",
        # "text": f"Less similar (N={less_N}, {int(less_acc)}% correct)",
        "match_idxs": less_similar_idx.tolist(),
        "selected_idxs": less_sampled.tolist(),
        "images": np.array(full_fns)[less_sampled].tolist(),
        "similarity": all_sims[less_sampled].tolist(),
        "correct": correct[less_sampled].tolist(),
    }

    prompt_data["selected_results"] = [less_dict, more_dict]
    save_data(data)


@app.route("/update_slider", methods=["POST"])
def update_slider():
    request_data = request.get_json()
    slider_value = request_data["slider_value"]
    session["slider_value"] = slider_value
    print(f"Slider value updated to {session['slider_value']}")
    split_similarities(slider_value)
    data = get_data()
    data["slider_value"] = slider_value
    save_data(data)
    return "Slider value updated successfully", 200


def generate_dsprites_metrics():
    shape_names = ["ellipse", "square"]
    color_names = ["red", "blue"]
    all_fns = []
    all_probs = []
    labels, s_labels = [], []
    for shape, color in itertools.product(shape_names, color_names):
        y = shape_names.index(shape)
        s = color_names.index(color)
        N = 500 if y == s else 100
        all_fns += [f"colored_dsprites/{shape}-{color}/{i}.png" for i in range(N)]
        labels += [y] * N
        s_labels += [s] * N
        base_prob = 0.9 if y == s else 0.1
        probs = np.array([base_prob]).repeat(N)
        probs += np.random.normal(scale=0.15, size=N)
        probs = np.clip(probs, 0, 1)
        probs = np.stack([1 - probs, probs], axis=1)
        all_probs.append(probs)
    labels = np.array(labels).astype(int)
    s_labels = np.array(s_labels).astype(int)

    metrics = dict()
    metrics["probs"] = np.concatenate(all_probs)
    metrics["preds"] = metrics["probs"].argmax(-1)
    corrects = metrics["probs"].argmax(-1) == labels
    metrics["acc"] = corrects.mean().item()
    metrics["classwise_accs"] = [corrects[labels == i].mean().item() for i in range(2)]
    metrics["labels"] = labels.tolist()
    metrics["s_labels"] = s_labels.tolist()
    metrics["filenames"] = all_fns
    return metrics


@app.route("/init_data", methods=["POST"])
def init_data():
    cleanup_old_sessions()
    dataset_name = session.get("dataset_name", default=default_dataset_name)
    # data = get_data()
    # if data is None:
    #     print("No data found; retrieving training results...")
    data = defaultdict(list)
    if dataset_name == "dsprites":
        metrics = generate_dsprites_metrics()
        data["metrics"] = metrics
        data["selected_class"] = "square"
        data["class_names"] = ["ellipse", "square"]
    else:
        nets = get_nets(dataset_name)
        outputs, full_fns = load_dataset(dataset_name)
        metrics = get_classwise_metrics(nets, outputs)
        data["metrics"] = metrics
        class_names = list(dataset_info[dataset_name]["idx_to_class"].values())
        classwise_accs = metrics["classwise_accs"]
        worst_class_idx = np.argmin(classwise_accs)
        data["selected_class"] = class_names[worst_class_idx]

        data["class_names"] = class_names
    data["dataset_index"] = dataset_sequence.index(dataset_name)
    data["all_prompts"] = []
    save_data(data)

    selected_results_text = [r["text"] for r in data["selected_results"]]
    # check if any text has substring "similar"
    using_slider = any("similar" in text for text in selected_results_text)
    if not using_slider:
        print("Not using slider; re-sampling.")
        sample_images()

    data = get_data()
    print([r["text"] for r in data["selected_results"]])
    return jsonify(data)


@app.route("/show_initial_data", methods=["POST"])
def show_initial_data():
    data = get_data()
    data["show_initial_data"] = True
    return jsonify(data)


@app.route("/continue", methods=["POST"])
def next_task():
    print(f"Submitting and moving on to the next task...")
    request_data = request.get_json()
    prolific_id = request_data["prolificID"]
    data = get_data()

    dataset = session.get("dataset_name", default=default_dataset_name)
    task_str = str(dataset)
    for prompt in data["all_prompts"]:
        prompt_data = data["prompt_data"][prompt]
        cutoff = prompt_data["slider_value"]
        score = prompt_data["prompt_score"]
        task_str += f"\n{prompt}\t{cutoff}\t{score}"

    log_dir = "logs/prolific"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/{prolific_id}.txt", "a") as f:
        sid = session["session_id"]
        time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{sid}\n")
        f.write(f"{time}\n")
        f.write(f"{task_str}\n")

    if dataset == "celeba":
        return jsonify({"out_of_tasks": True})

    current_idx = dataset_sequence.index(dataset)
    new_dataset = dataset_sequence[(current_idx + 1) % len(dataset_sequence)]
    session["dataset_name"] = new_dataset

    session["prompts"] = []
    session_id = session["session_id"]
    shutil.rmtree(f"session_data/{session_id}", ignore_errors=True)
    return jsonify({"out_of_tasks": False})


@app.route("/logData", methods=["POST"])
def log_data():
    data = request.get_json()
    prolific_id = data["prolificID"]
    output = data["data"]
    output["time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    log_dir = "logs/prolific"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/{prolific_id}.txt", "a") as f:
        f.write(json.dumps(output))
    return "", 200


@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())  # Generate a UUID as the session ID
    os.makedirs(f"session_data/{session['session_id']}", exist_ok=True)

    dataset_name = session.get("dataset_name", default=default_dataset_name)
    return render_template("index.html", data=data, zip=zip, dataset_name=dataset_name)


if __name__ == "__main__":
    port = 5002
    print(f"Starting Prolific server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=True)
