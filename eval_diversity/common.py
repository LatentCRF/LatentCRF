import urllib.request

import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import torch
from PIL import Image

MODEL_NAMES = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}


def weight_K(K, p=None):
    if p is None:
        return K / K.shape[0]
    else:
        return K * np.outer(np.sqrt(p), np.sqrt(p))


def normalize_K(K):
    d = np.sqrt(np.diagonal(K))
    return K / np.outer(d, d)


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_**q).sum()) / (1 - q)


def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def get_parti_prompts():
    url = "https://raw.githubusercontent.com/google-research/parti/5a657978134374ce28973948331b319adef164bd/PartiPrompts.tsv"

    with urllib.request.urlopen(url) as f:
        df = pd.read_csv(f, sep="\t")
    return df


def vis_image_list(im_list):
    image_arr_list = [np.array(im) for im in im_list]
    image_combined = np.concatenate(image_arr_list, axis=1)
    return Image.fromarray(image_combined)


def get_image_list(pipe_func, prompt, count=16, noise_list=None):
    image_list = []
    for i in range(count):
        if noise_list is not None:
            noise = noise_list[i]
        else:
            noise = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        image = pipe_func(prompt=prompt, latents=noise)
        image_list.append(image)
    return image_list
