# %%

import requests
from io import BytesIO
from PIL import Image
import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from  flax.serialization import to_bytes, from_bytes
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from diffusers import FlaxStableDiffusionImg2ImgPipeline
from diffusers import FlaxStableDiffusionInstructPix2PixPipeline
from diffusers import FlaxStableDiffusionImg2ImgPipeline

#from jax.experimental.compilation_cache import compilation_cache as cc
#cc.set_cache_dir("/tmp/sd_cache")

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

# pipeline, params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    #'../flax_models/instruct-pix2pix',
    '../flax_models/stable-diffusion-v1-5',
    # './instruct-pix2pix-model', 
    dtype=jnp.bfloat16,
    safety_checker=None
)
# %%


ema_params_path = './instruct-pix2pix-model/ema_params/ema_params.msgpack'

# Load the EMA parameters from disk
with open(ema_params_path, 'rb') as f:
    ema_params = from_bytes(params['unet'], f.read()) # template object (here params['unet']) needs to be isomorphic to the target object

popped_item = params.pop('unet', None)

# Update the pipeline parameters with the loaded EMA parameters
params['unet'] = ema_params


# %%

# Cyborg 
url = 'https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg'
image = download_image(url).resize((512, 512))
prompt = 'turn him into cyborg'
prompt = 'Generate a cartoonized version of the image'

# %% 
# Snowy mountains
url = 'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'
image = download_image(url).resize((512, 512))
prompt = 'make the mountains snowy'
prompt = 'Generate a cartoonized version of the image'

# %% 

# Run the pipeline

rng = create_key(1371)
num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompt] * num_samples, image=[image] * num_samples
)

p_params = replicate(params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)

output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    num_inference_steps=50,
    height=512,
    width=512,
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))

# Grid plot images
from diffusers.utils import make_image_grid
make_image_grid(output_images, rows=len(output_images)//4, cols=4)


# %%
import torch  
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import FlaxStableDiffusionInstructPix2PixPipeline

model_id = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
     model_id, 
    torch_dtype=torch.float32, 
    safety_checker=None
)


images = pipe(prompt, image=image).images
images[0].show()



# %%
