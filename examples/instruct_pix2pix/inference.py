# %%

import requests
from io import BytesIO
import PIL
from PIL import Image
import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from  flax.serialization import to_bytes, from_bytes
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from diffusers.utils import make_image_grid
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

pipeline, pipeline_params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
    # '../flax_models/instruct-pix2pix',
    # '../flax_models/stable-diffusion-v1-5',
    './instruct-pix2pix-model', 
    dtype=jnp.bfloat16,
    safety_checker=None
)

LOAD_NON_EMA_PARAMS = False # change if testing non-ema params

if LOAD_NON_EMA_PARAMS:
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

rng = create_key(1371)
num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
p_params = replicate(pipeline_params)

def run_pipeline(prompt, image):
    prompt_ids, processed_image = pipeline.prepare_inputs(
        prompt=[prompt] * num_samples, image=[image] * num_samples
    )

    prompt_ids = shard(prompt_ids)
    processed_image = shard(processed_image)

    output = pipeline(
        prompt_ids=prompt_ids,
        image=processed_image,
        params=p_params,
        prng_seed=rng,
        num_inference_steps=50,
        guidance_scale=7.5,
        image_guidance_scale=1.2,
        height=512,
        width=512,
        jit=True, # include for img2img
    ).images

    output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))

    # Grid plot images
    return make_image_grid(output_images, rows=len(output_images)//4, cols=4)


text_prompts = [
    "Generate a cartoonized version of the image",
    "turn him into a cyborg",
    "wipe out the lake",
    "make the mountains snowy",
    "Swap sunflowers with roses",
]
image_prompts = [
    "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png",
    "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png",
    "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg",
    "https://wehco.media.clients.ellingtoncms.com/img/photos/2019/06/25/vangogh1_t800.png?90232451fbcadccc64a17de7521d859a8f88077d",
]

images = list(map(lambda x : download_image(x).resize((512,512)), image_prompts))



# %%
# cartoonized (0), cyborg(1), wipe out lake (2), snowy (3), swap with roses (4)
# painting (0), mountains(1), sculpture (2), gogh(3) sunflowers(4)

M, N = 1, 2 # text, image
run_pipeline(text_prompts[M], images[N])


# %% 
# Cyborg 
url = 'https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg'
image = download_image(url).resize((512, 512))
prompt = 'turn him into a cyborg'
prompt = 'Generate a cartoonized version of the image'

# %% 
# Snowy mountains
url='https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'
url='https://cdn-lfs.huggingface.co/repos/f6/ea/f6ea2d9b15ffdf0b3d41d9f1adcc2056323a844b3c37335563295a4ccd8bbe3d/7405e2013907463cb6e0c1a15bab847b3d2f982ec4bd8f33610962cd23c87624?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27mountain.png%3B+filename%3D%22mountain.png%22%3B&response-content-type=image%2Fpng&Expires=1714197929&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxNDE5NzkyOX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9mNi9lYS9mNmVhMmQ5YjE1ZmZkZjBiM2Q0MWQ5ZjFhZGNjMjA1NjMyM2E4NDRiM2MzNzMzNTU2MzI5NWE0Y2NkOGJiZTNkLzc0MDVlMjAxMzkwNzQ2M2NiNmUwYzFhMTViYWI4NDdiM2QyZjk4MmVjNGJkOGYzMzYxMDk2MmNkMjNjODc2MjQ%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=G4E-9YqrNqqAR1Vcn3bnxy9KX4vopcFs9xMltX-8nb%7E0bRFxov-IL0nj-i2rvU0Ijxdx4foLXy0oQdysQQf2Dn%7ESiEVX8ONxy%7E9h-bsmsGscfo9cusZPZMpA5BUahtko-480cJApwJ39ohLhxRXQ7%7EqFCerCZ9qRZpV2hV87To%7EyHb7CGOjMKlxIgD1Wr103cTLmEKi8Bwc-FyeO1Nu9AA0ryOBjQgpkmKjFwup71TkGJHS3pmhG%7EmyD1mSx2iWSK3NQlnMce6-%7E1yQzp3N1go4t%7EqIfq0qxcAKOyGoVPjn5-O-s52xGQYZ%7EDjFfKP9uGA0zKMoC3V7B389-SiMptQ__&Key-Pair-Id=KVTP0A1DKRTAX'
image = download_image(url).resize((512, 512))
prompt = 'make the mountains snowy'
prompt = 'Generate a cartoonized version of the image'

# %% 
# Sunflowers
url='https://wehco.media.clients.ellingtoncms.com/img/photos/2019/06/25/vangogh1_t800.png?90232451fbcadccc64a17de7521d859a8f88077d'
image = download_image(url).resize((512, 512))
prompt = 'Swap sunflowers with roses'
prompt = 'Generate a cartoonized version of the image'

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
    guidance_scale=7.5,
    image_guidance_scale=1.2,
    height=512,
    width=512,
    jit=True, # include for img2img
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))

# Grid plot images
from diffusers.utils import make_image_grid
make_image_grid(output_images, rows=len(output_images)//4, cols=4)






# %%

# Multi-prompt inference

text_prompts = [
    "wipe out the lake",
    "make the mountains snowy",
    "Generate a cartoonized version of the image",
    "turn him into a cyborg",
    "Generate a cartoonized version of the image",
    "Swap sunflowers with roses",
    "Generate a cartoonized version of the image",
]
image_prompts = [
    "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png",
    "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png",
    "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png",
    "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg",
    "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg",
    "https://wehco.media.clients.ellingtoncms.com/img/photos/2019/06/25/vangogh1_t800.png?90232451fbcadccc64a17de7521d859a8f88077d",
    "https://wehco.media.clients.ellingtoncms.com/img/photos/2019/06/25/vangogh1_t800.png?90232451fbcadccc64a17de7521d859a8f88077d",
]



# %% 
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


import matplotlib.pyplot as plt

def show_images(images):
    # Plot images in rows of four (4)
    rows = len(images) // 4
    for i in range(rows):
        plt.figure(figsize=(20, 20))
        for j in range(4):
            plt.subplot(1, 4, j + 1)
            plt.imshow(images[i * 4 + j])
            plt.axis('off')
        plt.show()


def run_inference(
    imageprompts, 
    text_prompts, 
    num_inference_steps=50,
    image_guidance_scale=1.2,
    guidance_scale=7.5,
    height=512,
    width=512,
    jit=True,
    ):

    rng = create_key(1371)
    num_samples = jax.device_count()
    rng = jax.random.split(rng, jax.device_count())

    image_prompts = []
    # Download images if necessary and resize to 512x512
    for i, image in enumerate(imageprompts):
        if isinstance(image, PIL.Image.Image):
            image = image.resize((512, 512))
        elif isinstance(image, str):
            image = download_image(image).resize((512, 512))
        image_prompts.append(image)


    prompt_ids, processed_images  = [], []

    # Pad prompts so that they are a multiple of the number of TPU devices
    if len(text_prompts) < num_samples or (len(image_prompts) % num_samples != 0):
        multiple = int(np.ceil(len(text_prompts)/ num_samples))
        # replicate the shortfalls
        text_prompts = text_prompts * multiple * num_samples
        image_prompts = image_prompts * multiple * num_samples
        text_prompts = text_prompts[:multiple * num_samples]
        image_prompts = image_prompts[:multiple * num_samples]

    # Preprocess (tokenize) the prompts and images one at a time
    while len(text_prompts) > 0:
        prompt, image = text_prompts.pop(), image_prompts.pop()
        prompt_id, processed_image = pipeline.prepare_inputs(
            prompt=prompt , image=image
        )
        prompt_ids.append(prompt_id)
        processed_images.append(processed_image)

    # Convert to numpy arrays and remove the singleton dimension (N,1,77 etc.)
    ids = np.array(prompt_ids).squeeze(1)
    imgs = np.array(processed_images).squeeze(1)

    print(ids.shape, imgs.shape)
    p_params = replicate(pipeline_params)

    # Now each device gets a individual prompt and image if N > num_samples
    prompt_ids = shard(ids)
    processed_image = shard(imgs)
    print(prompt_ids.shape, processed_image.shape)

    output = pipeline(
        prompt_ids=prompt_ids,
        image=processed_image,
        params=p_params,
        prng_seed=rng,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        height=height,
        width=width,
        jit=jit, # include for img2img
    ).images

    # As the output returns in batches of num_devices, reshape the output  (N, H, W, C) ->(N, H, W, C)
    #output_images = jnp.einsum('...hwc->...chw', output)    

    H,W,C = output.shape[-3:]
    output_images = pipeline.numpy_to_pil(np.asarray(output.reshape(-1, H, W, C)))
    return output_images

images = run_inference(image_prompts, text_prompts)

# ... and display the results
show_images(images)

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
