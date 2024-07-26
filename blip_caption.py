import os
import pickle
import textwrap

import matplotlib.pyplot as plt
import torch
import yake
from keybert import KeyBERT
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from datasets_all import dataset_info

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def extract_caption(image_fn):
    raw_image = Image.open(image_fn).convert("RGB")
    text = "an image of"
    inputs = processor(raw_image, text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


def get_captions(dataset_name, split):
    output_filename = f"features/captions_{dataset_name}_{split}.pkl"
    if os.path.exists(output_filename):
        with open(output_filename, "rb") as f:
            all_captions = pickle.load(f)
        print(f"Loaded {len(all_captions)} captions for {dataset_name} {split}")
        return all_captions
    else:
        print(f"Saving captions for {dataset_name} {split}")
        data_obj = dataset_info[dataset_name]["data_obj"]
        dataset = data_obj(split=split, transform=transforms.ToTensor())
        image_fns = dataset.filename_array
        data_root = dataset_info[dataset_name]["data_root"]
        image_fns = [os.path.join(data_root, image_fn) for image_fn in image_fns]

        all_captions = []
        for image_fn in tqdm(image_fns):
            caption = extract_caption(image_fn)
            all_captions.append(caption)
            print(caption)

        with open(output_filename, "wb") as f:
            pickle.dump(all_captions, f)
        return all_captions


def display_images_with_captions(image_fns):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for ax, img_fn in zip(axes, image_fns):
        img = Image.open(img_fn)
        img = img.resize((224, 224))
        caption = extract_caption(img_fn)

        wrapped_caption = textwrap.fill(caption, 30)  # Wrap the caption text

        ax.imshow(img)
        ax.set_title(wrapped_caption, fontsize=16)
        ax.axis("off")

    plt.subplots_adjust(hspace=0.6)  # Increase space between subplots
    plt.tight_layout()
    plt.show()


def extract_keyword(captions, max_size=3, num_keywords=30):
    captions = [caption.replace("an image of ", "") for caption in captions]
    captions_str = ", ".join(captions)

    kw_model = KeyBERT()
    keybert_outputs = kw_model.extract_keywords(
        captions_str,
        keyphrase_ngram_range=(1, max_size),
        top_n=num_keywords,
        nr_candidates=num_keywords * 2,
        stop_words=None,
    )
    keybert_outputs = [keyword[0] for keyword in keybert_outputs]

    custom_kw_extractor = yake.KeywordExtractor(
        lan="en", n=max_size, dedupLim=0.9, top=num_keywords, features=None
    )
    yake_outputs = custom_kw_extractor.extract_keywords(captions_str)
    yake_outputs = [keyword[0] for keyword in yake_outputs]

    return list(set(keybert_outputs + yake_outputs))


if __name__ == "__main__":
    captions = get_captions(dataset_name="imagenet-9", split="val")
    get_captions(dataset_name="waterbirds", split="val")
    get_captions(dataset_name="waterbirds", split="test")
    get_captions(dataset_name="celeba", split="val")
    get_captions(dataset_name="celeba", split="test")

    keywords = extract_keyword(captions)
    print(keywords)
