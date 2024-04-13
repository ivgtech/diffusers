# %%

import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline
from diffusers import FlaxStableDiffusionInstructPix2PixPipeline

from flax.core.frozen_dict import FrozenDict, unfreeze, freeze

from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("/tmp/sd_cache")

# %% 
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

url = 'https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg'
image = download_image(url).resize((512, 512))
prompt = "turn him into cyborg"


def create_key(seed=0):
    return jax.random.PRNGKey(seed)



pipeline, params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
  '../flax_models/instruct-pix2pix',
  # './instruct-pix2pix-model', 
    dtype=jnp.bfloat16,
    safety_checker=None
)

# %%
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
# %%


from diffusers.utils import make_image_grid
make_image_grid(output_images, rows=len(output_images)//4, cols=4)
# %%