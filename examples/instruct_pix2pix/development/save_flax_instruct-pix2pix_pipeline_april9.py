#  %% 

from flax_common_imports import *

# %%
# Load instruct-pix2pix weights from Hugging Face Hub and create Flax pipeline

MODEL_NAME= 'timbrooks/instruct-pix2pix'

dtype = jax.numpy.bfloat16

tokenizer = CLIPTokenizer.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    revision="flax",
    subfolder="tokenizer",
)

text_encoder = FlaxCLIPTextModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    revision="flax",
    subfolder="text_encoder",
)

vae, vae_state = FlaxAutoencoderKL.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    revision="flax",
    subfolder="vae",
)

# # Flax does not have a EulerAncestralDiscreteScheduler
# scheduler, scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
#     'timbrooks/instruct-pix2pix',
#     from_pt=True,
#     revision='main', # pytorch revision
#     subfolder='scheduler',
# )

scheduler = FlaxPNDMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
)

unet, unet_state = FlaxUNet2DConditionModel.from_pretrained(
    'timbrooks/instruct-pix2pix',
    from_pt=True,
    revision='main', # pytorch revision
    subfolder='unet',
)

safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)

# Create a Flax wrapper pipeline class 
pipeline = FlaxStableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer, # FlaxPreTrainedModel.save_pretrained of PreTrainedTokenizer
    scheduler=scheduler, # FlaxSchedulerMixin.save_pretrained of FlaxEulerDiscreteScheduler 
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)

# %% 
# Save flax wrapper to disk as a local pipeline

assert pipeline.unet.in_channels == 8, 'Expected the Instruct-Pix2Pix model to have 8 input channels instead'

text_encoder_state = text_encoder._params
outdir = '/home/v/diffusers/examples/flax_models/instruct-pix2pix'

if not os.path.exists(outdir):

    pipeline.save_pretrained(
        outdir,
        params={
            "text_encoder": text_encoder_state,  # FlaxPreTrainedModel.save_pretrained
            "vae": vae_state,
            "unet": unet_state,
            "safety_checker": safety_checker.params,
        },
    )

else:
    print(f"Directory {outdir} already exists, skipping save")

####################################################################################################################################

# %%
# Test the saved pipeline by running inference

# del pipeline, text_encoder, vae, unet, safety_checker, scheduler, tokenizer, text_encoder_state, vae_state, unet_state, scheduler_state

dtype = jax.numpy.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    '../flax_models/instruct-pix2pix',
    revision='flax',
    dtype=dtype,
)

vae = pipeline.vae
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer
unet = pipeline.unet
scheduler = pipeline.scheduler
safety_checker = pipeline.safety_checker
feature_extractor = pipeline.feature_extractor
dtype = pipeline.dtype


# %%
# Helper functions

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)

def preprocess(image, dtype):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2.0 * image - 1.0

def prepare_inputs(
    prompt: Union[str, List[str]], 
    image: Union[Image.Image,  List[Image.Image]], 
    negative_prompt: Union[str, List[str], None], 
    tokenizer: CLIPTokenizer
    ):
    if not isinstance(prompt, (str, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if not isinstance(negative_prompt, (str, list)):
        raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
    if not isinstance(image, (Image.Image, list)):
        raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")
    if isinstance(image, Image.Image):
        image = [image]

    processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    neg_text_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    return text_input.input_ids, processed_images, neg_text_input.input_ids

def numpy_to_pil(images):
    #  from src/diffusers/pipelines/pipeline_flax_utils.py
    if images.ndim == 3: images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1: # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image  

    
    
    
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: jnp.ndarray, generator: Optional[jax.random.PRNGKey] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # return encoder_output.latent_dist.sample(generator)
        return jax.random.categorical(generator, encoder_output.latent_dist.logits).astype(jnp.float32)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# convert helper functions to jax.numpy
def prepare_image_latents(
     image, batch_size, num_images_per_prompt, dtype, do_classifier_free_guidance ):
    # convert to jax.numpy
    if not isinstance(image, (jax.numpy.ndarray, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        image_latents = image
    else:
        image_latents = retrieve_latents(vae.encode(image), sample_mode="argmax")

    if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
        # expand image_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // image_latents.shape[0]
        image_latents = jnp.concatenate([image_latents] * additional_image_per_prompt, axis=0)
    elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        image_latents = jnp.concatenate([image_latents], axis=0)

    if do_classifier_free_guidance:
        uncond_image_latents = jnp.zeros_like(image_latents)
        image_latents = jnp.concatenate([image_latents, image_latents, uncond_image_latents], axis=0)

    return image_latents



def encode_prompt(
    prompt_batch: List[str],
    text_encoders: List[FlaxCLIPTextModel],
    tokenizers: List[CLIPTokenizer],
    proportion_empty_prompts: float,
    rng: jax.random.PRNGKey,
    is_train: bool = True
) -> (jnp.ndarray, jnp.ndarray):
    prompt_embeds_list = []
    captions = []
    rng, sub_rng = jax.random.split(rng)

    for caption in prompt_batch:
        if jax.random.uniform(sub_rng) < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # split rng again for random choices
            rng, choice_rng = jax.random.split(rng)
            chosen_index = jax.random.randint(choice_rng, (1,), 0, len(caption))
            captions.append(caption[chosen_index[0]] if is_train else caption[0])

    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="jax"
        )
        text_input_ids = text_inputs.input_ids
        # pass the text_encoder params to the text_encoder explicitly
        outputs = text_encoder(text_input_ids, params=params["text_encoder"])
        prompt_embeds = outputs.last_hidden_state

        # handling pooled outputs is model-specific, here we assume last_hidden_state for simplicity
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = prompt_embeds[:, 0, :]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = jnp.concatenate(prompt_embeds_list, axis=-1)
    # For pooled prompt embeddings: here we assume the first token's embedding is used as the pooled representation
    pooled_prompt_embeds = prompt_embeds[:, 0, :].reshape(prompt_embeds.shape[0], -1)
    
    return prompt_embeds, pooled_prompt_embeds


from torch.utils.data import DataLoader

def torch_to_numpy(batch):
    """
    Convert a batch of PyTorch tensors to NumPy arrays.
    """
    return {k: v.numpy() for k, v in batch.items()}

class JAXDataLoader:
    """
    training_generator = JAXDataLoader(train_dataloader_torch)

    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in training_generator:
            images, labels = batch["images"], batch["labels"]
            labels = one_hot(labels, n_targets)  # Convert labels to one-hot encoding
            params = update(params, images, labels)  # Update model parameters
        epoch_time = time.time() - start_time
        # Compute accuracy on training and test datasets
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            # Convert PyTorch tensors in the batch to NumPy arrays
            yield torch_to_numpy(batch)

"""
import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

"""


# %%
# encode_prompt function tests

rng = jax.random.PRNGKey(0) 
prompt_batch = ["A photo of a dog in the park", "A painting of a sunset"] 
# Encode the prompts
prompt_embeds, pooled_prompt_embeds = encode_prompt(
    prompt_batch, [text_encoder], [tokenizer], proportion_empty_prompts=0.5, rng=rng, is_train=True
)

print(prompt_embeds.shape)
print(pooled_prompt_embeds.shape)

from preprocess_load_data_april4 import  train_dataloader_torch, plot_batch
#plot_batch(train_dataloader, pipeline.tokenizer)

batch = next(iter(train_dataloader_torch))
prompt_batch = batch["input_ids"]

prompt_embeds, pooled_prompt_embeds = encode_prompt(
    prompt_batch, [text_encoder], [tokenizer], proportion_empty_prompts=0.5, rng=rng, is_train=True 
    )
print(prompt_embeds.shape)
print(pooled_prompt_embeds.shape)

# %%

# %% 
# Instruction-based image-to-image editing 


url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
prompts = "A fantasy landscape, trending on artstation"
neg_prompts = "not a fantasy landscape, trending on artstation" 


url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
prompts = "wipe out the lake"
neg_prompts = ""
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

# edited_image = pipe(prompt, 
#     image=image, 
#     num_inference_steps=num_inference_steps, 
#     image_guidance_scale=image_guidance_scale, 
#     guidance_scale=guidance_scale,
#     generator=generator,
# ).images[0]

# edited_image.save("edited_image.png")
# %%



# %% 
# Inference function run as part of the AOT compilation

def my_generate(
    prompt_ids: jnp.ndarray,
    image: jnp.ndarray,
    params: Union[Dict, FrozenDict],
    prng_seed: jax.Array,
    start_timestep: int,
    num_inference_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    noise: Optional[jnp.ndarray] = None,
    neg_prompt_ids: Optional[jnp.ndarray] = None,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # get prompt text embeddings
    prompt_embeds = text_encoder(prompt_ids, params=params["text_encoder"])[0]

    # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
    # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
    batch_size = prompt_ids.shape[0]

    max_length = prompt_ids.shape[-1]

    if neg_prompt_ids is None:
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
        ).input_ids
    else:
        uncond_input = neg_prompt_ids
    negative_prompt_embeds = text_encoder(uncond_input, params=params["text_encoder"])[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

    # Ensure model output will be `float32` before going into the scheduler
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

    latents_shape = (
        batch_size,
        unet.config.in_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    if noise is None:
        noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
    else:
        if noise.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {noise.shape}, expected {latents_shape}")

    # Create init_latents
    init_latent_dist = vae.apply({"params": params["vae"]}, image, method=vae.encode).latent_dist
    init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
    init_latents = vae.config.scaling_factor * init_latents

    def loop_body(step, args):
        latents, scheduler_state = args
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes

        # Expand the latents if we are doing classifier free guidance.
        # The latents are expanded 3 times because for pix2pix the guidance\
        # is applied for both the text and the input image.
        latents_input = jnp.concatenate([latents] * 2 ) # 3)
        

        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])

        # concat latents, init_latents(image_latents) in the channel dimension
        latents_input = jnp.concatenate([latents, init_latents], axis=1)

        latents_input = scheduler.scale_model_input(scheduler_state, latents_input, t)

        # predict the noise residual
        noise_pred = unet.apply(
            {"params": params["unet"]},
            jnp.array(latents_input),
            jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context,
        ).sample

        # perform guidance
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        return latents, scheduler_state

    # set the scheduler state
    scheduler_state = scheduler.set_timesteps(
        params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
    )

    latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)
    latents = scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep) # TODO: exception here

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * params["scheduler"].init_noise_sigma
    if DEBUG:
        # run with python for loop
        for i in range(start_timestep, num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        latents, _ = jax.lax.fori_loop(start_timestep, num_inference_steps, loop_body, (latents, scheduler_state))

    # scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.apply({"params": params["vae"]}, latents, method=vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image




# %%
# Ahead of time compilation (AOT)

# Replicate mutable arrays across devices before passing them to the AOT compiled function
def replicate_all(prompt_ids, image, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    p_image_ids = replicate(image)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_image_ids, p_neg_prompt_ids, rng

# Replicate all model state across devices before passing them to the AOT compiled function
p_params = replicate(params)

def get_timestep_start( num_inference_steps, strength): # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    return max(num_inference_steps - init_timestep, 0)

def aot_compile(
    prompt=prompts,
    negative_prompt=neg_prompts,
    image=init_img,
    seed=0,
    guidance_scale=7.5,
    strength=0.75,
    num_inference_steps=50,
    height=512,
    width=768,
):

    prompt_ids, image_ids, neg_prompt_ids = prepare_inputs(prompt, image, negative_prompt, tokenizer=pipeline.tokenizer)
    prompt_ids, image_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, image_ids, neg_prompt_ids, seed)
    start_timestep = get_timestep_start(num_inference_steps, strength)

    if isinstance(guidance_scale, float):
        # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
        # shape information, as they may be sharded (when `jit` is `True`), or not.
        guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
        if len(prompt_ids.shape) > 2:
            # Assume sharded
            guidance_scale = guidance_scale[:, None]

    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    # Static argnums are pipe, start_timestep, num_inference_steps, height, width
    # Non-static (0) args are (sharded) input tensors mapped over their first dimension : prompt_ids, image, params, prng_seed,...,guidance_scale, noise, neg_prompt_ids,
    #in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0, 0),
    # static_broadcasted_argnums=(0, 5, 6, 7, 8),)
    #(pipe, prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,):

    #in_axes=(0, 0, 0, 0, None, None, None, None, 0, 0, 0), 
    # static_broadcasted_argnums=(5, 6, 7, 8),
    return (
        # pmap(pipeline._generate, static_broadcasted_argnums=[4,5, 6, 7]) # (5, 6, 7, 8) are the static args
        # NOTE: generate is a functional version of pipeline._generate
        pmap(my_generate, static_broadcasted_argnums=[4,5, 6, 7]) # (5, 6, 7, 8) are the static args
        .lower(
                prompt_ids,
                image_ids,
                p_params, 
                rng, 
                start_timestep, # static (4)  
                num_inference_steps,  # static
                height,  # static
                width,  # static
                g, 
                None, # noise
                neg_prompt_ids 
            )
        .compile()
    )

# %%
# Initial compilation 
start = time.time()

p_generate = aot_compile()

print(f"Inference in {time.time() - start}")
# 102.0s if not cc cached, otherwise >12s

# %%

# Here we use a curried function to avoid recompiling the calling function every time we want to generate an image
def generate(prompt, image, negative_prompt, seed=0, guidance_scale=7.5):
    prompt_ids, image_ids, neg_prompt_ids = prepare_inputs(prompt, image, negative_prompt, tokenizer=pipeline.tokenizer)
    prompt_ids, image_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, image_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(prompt_ids, image_ids, p_params, rng, g, None, neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return numpy_to_pil(np.array(images))

# %% 
# Middle of the road inference time
start = time.time()
images = generate(prompts, init_img, neg_prompts)
images[0].show()
print(f"Inference in {time.time() - start}")
# ~3.05

# %%
# Compiled function is now fully cached and so inference is at optimal speeds
start = time.time()
images = generate(prompts, init_img, neg_prompts)
images[2].show()
print(f"Inference in {time.time() - start}")
# ~2.71s 

# %%
# Plot all images
make_image_grid(images, rows=len(images)//4, cols=4)

# %%
