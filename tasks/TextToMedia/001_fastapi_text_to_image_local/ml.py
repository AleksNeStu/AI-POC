from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
import os

# Token https://huggingface.co/settings/tokens
API_KEY_HUGGING_FACE = os.environ.get("API_KEY_HUGGING_FACE")
token_path = Path("token.txt")
token = token_path.read_text().strip() or API_KEY_HUGGING_FACE

# Device (Cuda GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    # https://huggingface.co/models?library=diffusers&sort=downloads
    # https://huggingface.co/runwayml/stable-diffusion-v1-5
    # https://huggingface.co/blog/stable_diffusion
    # Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from
    # CompVis, Stability AI and LAION. It is trained on 512x512 images from a subset of the LAION-5B database. LAION-5B is the largest, freely accessible multi-modal dataset that currently exists.
    # pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    # revision="fp16",
    variant="fp16",
    # Remove the torch_dtype=torch.float16 argument: If you don't need half-precision computations, you can simply remove this argument. PyTorch will then use torch.float32 by default, which is widely supported on both CPUs and GPUs.
    # torch_dtype=torch.float16,
    # Keyword arguments {'use_auth_token': '***'} are not expected by
    # use_auth_token=token,
)
pipe.to(device)


def gen_image_wo_api(prompt: str = None):
    prompt = prompt or "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    image.save("astronaut_rides_horse.png")


def obtain_image(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,  # NOTE: Use less e.g. 3 in debug mode to test whole app
    guidance_scale: float = 7.5,
) -> Image:
    # If you want deterministic output you can seed a random seed and pass a generator to the pipeline
    generator = None if seed is None else torch.Generator(device).manual_seed(seed)
    print(f"Using device: {pipe.device}")
    # In general, results are better the more steps you use, however the more steps, the longer the generation takes.
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image


# image = obtain_image(prompt, num_inference_steps=5, seed=1024)
