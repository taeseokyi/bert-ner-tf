import torch
from torch import autocast, seed
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"

prompt = 'Pilea peperoma'
with autocast("cuda"):
  image = pipe(prompt, guidance_scale=7.5, width=904, height=768, seed=1256).images[0]

image.save("astronaut_rides_horse.png")
