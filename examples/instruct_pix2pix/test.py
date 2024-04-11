# %%
                                                                # Flax inference
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline

def create_key(seed=0):
    return jax.random.PRNGKey(seed)


rng = create_key(0)

dtype = jnp.bfloat16
# load the pipeline
pipeline, params  = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="flax",
    dtype=dtype,
)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompts = "A fantasy landscape, trending on artstation"

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompts] * num_samples, image=[init_image] * num_samples
)
p_params = replicate(params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)




# %% 
output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    strength=0.75,
    num_inference_steps=50,
    jit=True,
    height=512,
    width=768,
).images

# 2nd init_latents.shape, noise.shape, latents_shape (1, 4, 64, 96) (1, 4, 64, 96) (1, 4, 64, 96)

# (1, 4, 64, 96) (1, 4, 64, 96) (1, 4, 64, 96)
# 2nd init_latents.shape, noise.shape, latents_shape (1, 4, 64, 96) (1, 4, 64, 96) (1, 4, 64, 96)
# 3rd latents_input.shape, scheduler_state.timesteps (2, 4, 64, 96) Traced<ShapedArray(int32[51])>with<DynamicJaxprTrace(level=1/1)>
# 4th t.shape, timestep.shape () (2,)
# 5th latents_input.shape (2, 4, 64, 96)


output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
# %%
output_images[0]
# %%
