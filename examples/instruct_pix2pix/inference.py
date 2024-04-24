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

from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("/tmp/sd_cache")


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def create_key(seed=0):
    return jax.random.PRNGKey(seed)


# pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
pipeline, pipeline_params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
    # '../flax_models/instruct-pix2pix',
    # '../flax_models/stable-diffusion-v1-5',
    './instruct-pix2pix-model', 
    dtype=jnp.bfloat16,
    safety_checker=None
)

# %%
load_non_ema_params = False # change if testing non-ema params

if load_non_ema_params:
    params_path = './instruct-pix2pix-model/unet/non_ema.msgpack'

    # Load the EMA parameters from disk
    with open(params_path, 'rb') as f:
        non_ema_params = from_bytes(pipeline_params['unet'], f.read()) # template object (here params['unet']) needs to be isomorphic to the target object

    # Optionally, remove the 'unet' key from the pipeline parameters
    popped_item = pipeline_params.pop('unet', None)

    # Update the pipeline params 
    pipeline_params['non_ema'] = non_ema_params

# Otherwise, only the EMA parameters are saved to disk under the 'unet' key in the params dictionary

# %% 
# Snowy mountains
url='https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'
url='https://cdn-lfs.huggingface.co/repos/f6/ea/f6ea2d9b15ffdf0b3d41d9f1adcc2056323a844b3c37335563295a4ccd8bbe3d/7405e2013907463cb6e0c1a15bab847b3d2f982ec4bd8f33610962cd23c87624?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27mountain.png%3B+filename%3D%22mountain.png%22%3B&response-content-type=image%2Fpng&Expires=1714197929&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxNDE5NzkyOX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9mNi9lYS9mNmVhMmQ5YjE1ZmZkZjBiM2Q0MWQ5ZjFhZGNjMjA1NjMyM2E4NDRiM2MzNzMzNTU2MzI5NWE0Y2NkOGJiZTNkLzc0MDVlMjAxMzkwNzQ2M2NiNmUwYzFhMTViYWI4NDdiM2QyZjk4MmVjNGJkOGYzMzYxMDk2MmNkMjNjODc2MjQ%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=G4E-9YqrNqqAR1Vcn3bnxy9KX4vopcFs9xMltX-8nb%7E0bRFxov-IL0nj-i2rvU0Ijxdx4foLXy0oQdysQQf2Dn%7ESiEVX8ONxy%7E9h-bsmsGscfo9cusZPZMpA5BUahtko-480cJApwJ39ohLhxRXQ7%7EqFCerCZ9qRZpV2hV87To%7EyHb7CGOjMKlxIgD1Wr103cTLmEKi8Bwc-FyeO1Nu9AA0ryOBjQgpkmKjFwup71TkGJHS3pmhG%7EmyD1mSx2iWSK3NQlnMce6-%7E1yQzp3N1go4t%7EqIfq0qxcAKOyGoVPjn5-O-s52xGQYZ%7EDjFfKP9uGA0zKMoC3V7B389-SiMptQ__&Key-Pair-Id=KVTP0A1DKRTAX'
image = download_image(url).resize((512, 512))
prompt = 'Generate a cartoonized version of the image'
prompt = 'make the mountains snowy'

# %% 
# Sunflowers
url='https://wehco.media.clients.ellingtoncms.com/img/photos/2019/06/25/vangogh1_t800.png?90232451fbcadccc64a17de7521d859a8f88077d'
image = download_image(url).resize((512, 512))
prompt = 'Swap sunflowers with roses'

# %%

# Cyborg 
url = 'https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg'
image = download_image(url).resize((512, 512))
prompt = 'Generate a cartoonized version of the image'
prompt = 'turn him into cyborg'


# %% 
# Run the pipeline

rng = create_key(1371)
num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompt] * num_samples, image=[image] * num_samples
)

p_params = replicate(pipeline_params)
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
    jit=True, # include for img2img
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))

# Grid plot images
from diffusers.utils import make_image_grid
make_image_grid(output_images, rows=len(output_images)//4, cols=4)


# %%
# import torch  
# from diffusers import StableDiffusionInstructPix2PixPipeline
# from diffusers import FlaxStableDiffusionInstructPix2PixPipeline

# model_id = "timbrooks/instruct-pix2pix"

# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#      model_id, 
#     torch_dtype=torch.float32, 
#     safety_checker=None
# )


# images = pipe(prompt, image=image).images
# images[0].show()



# %%
