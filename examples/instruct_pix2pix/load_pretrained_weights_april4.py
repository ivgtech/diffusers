# %%

import os
import sys
import time
import requests
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
from jax import pmap
# Let's cache the model compilation, so that it doesn't take as long the next time around.
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("/tmp/sdxl_cache")
NUM_DEVICES = jax.device_count()

from flax.jax_utils import replicate
from flax.training.common_utils import shard
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict

import torch
from huggingface_hub import create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from transformers import FlaxCLIPTextModel, CLIPTokenizer

from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    WEIGHTS_NAME,
    PushToHubMixin,
)
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxDDPMScheduler
from diffusers.models.modeling_utils import load_state_dict 
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import FlaxCLIPTextModel, CLIPTokenizer
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxDDPMScheduler
from diffusers.models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from diffusers.models.modeling_utils import load_state_dict


# local imports
from flax_from_hf_pretrained_april4 import get_pretrained
from model_converter import load_from_standard_weights

def load_pretrained_weights(input_file: str, device: str) -> dict[str, torch.Tensor]:
    model = torch.load(input_file, map_location=device, weights_only = False)["state_dict"]
    return model

def download_pretrained_weights(model_url: str, download_dir: str, model_name: str) -> None:
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    try:
        if not os.path.exists(f"{download_dir}/{model_name}"):
            os.system(f"wget {model_url} -O {download_dir}/{model_name}")
            print(f"Model downloaded to {download_dir}/{model_name}")
        else:
            print(f"Model already exists at {download_dir}/{model_name}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)
      
# %%
# 
# (A) Initial setup
#       The Instruct pix2pix authors have a Pytorch checkpoint available on their GitHub repository, as well as a Huggingface Hub repository. To use either of these,
#       we need to convert the Pytorch models to Flax first. We will use the Huggingface Hub repository to download the Pytorch models, convert them to Flax, and save them to disk.
#       Afterward, we will load the Flax models from disk and use them for all experiments.
#       (1) Load the pretrained instruct-pix2pix models from Huggingface Hub
#       (2) Convert them from Pytorch to Flax
#       (3) Save the the new models and state dictionaries to disk
#       (4) Download and preprocess the dataset
# (B) Inference 
#       (4) Load the converted instruct-pix2pix models from disk
#       (5) Load the dataloader
#       (6) Train the models
# (C) Training 
# (D) Testing 

# %%
# Convert the instruct-pix2pix pretrained models from Pytorch to Flax

scheduler, schduler_params = FlaxDDPMScheduler.from_pretrained("timbrooks/instruct-pix2pix", subfolder="scheduler")
text_encoder = FlaxCLIPTextModel.from_pretrained("timbrooks/instruct-pix2pix", subfolder='text_encoder', dtype=jnp.bfloat16)
tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer', dtype=jnp.bfloat16)
vae, vae_params = get_pretrained("timbrooks/instruct-pix2pix", 'vae', FlaxAutoencoderKL)
unet, unet_params = get_pretrained("timbrooks/instruct-pix2pix", 'unet', FlaxUNet2DConditionModel)


 # %%
# Save the Flax unet, vae, text_encoder and scheduler models and their configuration files to a directory 
# so that they can be reloaded using the [`~FlaxModelMixin.from_pretrained`] class method.

unet.save_pretrained(params=unet_params,save_directory='../flax_models/unet' )
vae.save_pretrained(params=vae_params,save_directory='../flax_models/vae' )
scheduler.save_pretrained(params=schduler_params,save_directory='../flax_models/scheduler' )
text_encoder.save_pretrained(save_directory='../flax_models/text_encoder' )

# Either save the tokenizer files (without -r to avoid sym linking issues) or reload from the huggingface hub each time
os.makedirs("../flax_models/tokenizer", exist_ok=True)
os.system("cp ~/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix/snapshots/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d/tokenizer/* ../flax_models/tokenizer")    


##################################################################################################################################################################################
# setup complete
##################################################################################################################################################################################


# %%
# Load the pretrained Flax models from disk

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained('../flax_models/unet' )
vae, vae_params = FlaxAutoencoderKL.from_pretrained('../flax_models/vae' )  
scheduler, schduler_params = FlaxDDPMScheduler.from_pretrained('../flax_models/scheduler' )
text_encoder = FlaxCLIPTextModel.from_pretrained('../flax_models/text_encoder' )
text_encoder_params = text_encoder.params
tokenizer = CLIPTokenizer.from_pretrained('../flax_models/tokenizer' )



# %% 
# Load the dataset and data loader : https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples
# dict_keys(['input_image', 'edit_prompt', 'edited_image', 'original_pixel_values', 'edited_pixel_values', 'input_ids'])

# local import
from preprocess_load_data_april4 import train_dataloader, plot_batch
assert train_dataloader is not None, "Error: Dataset dataloader not loaded correctly"

for i in range(2):
  plot_batch(train_dataloader, tokenizer)



# %%
# Inference 

from transformers import FlaxCLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import  FlaxPNDMScheduler,  FlaxStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers import FlaxStableDiffusionInstructPix2PixPipeline

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


# Create the pipeline using using the trained modules and save it.
if jax.process_index() == 0:
    scheduler = FlaxPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
    )
    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", from_pt=True
    )
    pipeline = FlaxStableDiffusionInstructPix2PixPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    )

    pipeline.save_pretrained(
       'gatech/instruct-pix2pix', 
        params={
            "text_encoder": get_params_to_save(text_encoder_params),
            "vae": get_params_to_save(vae_params),
            "unet": get_params_to_save(unet_params),
            "safety_checker": safety_checker.params,
        },
    )
# %%
safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    from_pt=True
    )
safety_checker_params = safety_checker.params
safety_checker.save_pretrained(params=safety_checker_params, save_directory='gatech/instruct-pix2pix/safety_checker' )


# %%
from diffusers import FlaxStableDiffusionInstructPix2PixPipeline
dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
    'gatech/instruct-pix2pix',
    )    



# %%

image  = Image.open('/home/v/instruct-pix2pix/imgs/example.jpg')
prompt = 'turn him into cyborg'

# 2. We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

# 3. Next, we define the different inputs to the pipeline
default_prompt = prompt 
default_neg_prompt = ""
default_image = image
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
width = 1024
height = 1024


num_samples = jax.device_count()
rng = jax.random.PRNGKey(default_seed)
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompt] * num_samples, image=[image] * num_samples
)

p_params = replicate(params)
prompt_ids = shard(prompt_ids)
image_ids = shard(processed_image)

# %%

def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids


p_params = replicate(params)

def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng


def aot_compile(
    prompt_ids=prompt_ids,
    image_ids=image_ids,
    p_params=p_params,
    seed=default_seed,
    guidance_scale=default_guidance_scale,
    num_inference_steps=default_num_steps,
):
    #prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, image_ids, rng = replicate_all(prompt_ids, image_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]

    return (
        pmap(pipeline._generate, static_broadcasted_argnums=[3, 4, 5, 9])
        .lower(
            prompt_ids,
            image_ids,
            p_params,
            rng,
            num_inference_steps,  # num_inference_steps
            height,  # height
            width,  # width
            g,
            None,
            False,  # return_latents
        )
        .compile()
    )


start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")


def generate(prompt, negative_prompt, seed=default_seed, guidance_scale=default_guidance_scale):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(prompt_ids, image, p_params, rng, g, None, neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))


start = time.time()
neg_prompt = ""
images = generate(prompt, neg_prompt)
print(f"First inference in {time.time() - start}")

start = time.time()
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")

for i, image in enumerate(images):
    image.save(f"castle_{i}.png")



# %%
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import PIL
import requests
from io import BytesIO
from diffusers import FlaxStableDiffusionInpaintPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipeline, params = FlaxStableDiffusionInpaintPipeline.from_pretrained(
    "xvjiarui/stable-diffusion-2-inpainting"
)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
init_image = num_samples * [init_image]
mask_image = num_samples * [mask_image]
prompt_ids, processed_masked_images, processed_masks = pipeline.prepare_inputs(
    prompt, init_image, mask_image
)
# shard inputs and rng

params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)
processed_masked_images = shard(processed_masked_images)
processed_masks = shard(processed_masks)

images = pipeline(
    prompt_ids, processed_masks, processed_masked_images, params, prng_seed, num_inference_steps, jit=True
).images

# %%




def create_key(seed=0):
    return jax.random.PRNGKey(seed)


rng = create_key(0)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))

prompts = "A fantasy landscape, trending on artstation"

# pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
#         "CompVis/stable-diffusion-v1-4",
#         revision="flax",
#         dtype=jnp.bfloat16,
#)

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompts] * num_samples, image=[init_img] * num_samples
)
p_params = replicate(params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)

# output = pipeline(
#     prompt_ids=prompt_ids,
#     image=processed_image,
#     params=p_params,
#     prng_seed=rng,
#     strength=0.75,
#     num_inference_steps=50,
#     jit=True,
#     height=512,
#     width=768,
# ).images

# output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))


# # %%

# %%
