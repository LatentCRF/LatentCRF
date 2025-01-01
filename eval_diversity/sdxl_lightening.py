import clip
import common
import numpy as np
import torch
import tqdm
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Load Data.
df = common.get_parti_prompts()
print(df.head())
pos_text = df["Prompt"].values
print(pos_text[0])

# Load Model.
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)
model, preprocess = clip.load("ViT-B/32", device=device)


def pipe_func(prompt, latents):
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, latents=latents).images[0].resize((512, 512))
    return image


RUN_EVAL = True  # debug purposes

if RUN_EVAL:
    score_list = []
    pipe.set_progress_bar_config(disable=True)
    mean_score = 0
    progress_bar = tqdm.tqdm(enumerate(pos_text), total=len(pos_text), desc=f"Vendi Score: {mean_score:.2f}")

    for iter_idx, prompt in progress_bar:
        feature_list = []
        for i in range(16):
            noise = torch.randn(1, 4, 64, 64, dtype=torch.float16)
            with torch.no_grad():
                image = pipe_func(prompt=prompt, latents=noise)
                image = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
            feature_list.append(image_features)

        feature_list = torch.cat(feature_list, dim=0)
        similarity = feature_list @ feature_list.T
        score = common.score_K(similarity.cpu().numpy(), q=1, p=None, normalize=True)

        score_list.append(score)
        mean_score = np.mean(score_list)
        progress_bar.set_description(f"Vendi Score: {mean_score:.2f}")

    print(f"Final mean Vendi Score: {mean_score:.3f}")