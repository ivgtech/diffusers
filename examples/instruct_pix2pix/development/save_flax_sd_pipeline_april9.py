# %% 
from flax_common_imports import *

 # %%
 #                                                             Save SD pipeline
 ##############################################################################

outdir = '/home/v/diffusers/examples/flax_models/stable-diffusion-v1-5'

# save pipeline to disk
if not os.path.exists(outdir):
    print("Saving pipeline to disk ...")

    # load flax pipeline from from huggingface an non-EMA weights for the unet and vae models
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', 
        revision='flax',  # non-EMA weights (use 'bf16' for EMA weights)
        # from_pt=True, # not an option here as there is no .bin file for this revision
    )

    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=pipeline.text_encoder,
        vae=pipeline.vae,
        unet=pipeline.unet,
        tokenizer=pipeline.tokenizer,
        scheduler=pipeline.scheduler,
        safety_checker=pipeline.safety_checker,
        feature_extractor=pipeline.feature_extractor,
    )

    pipeline.save_pretrained(
        outdir,
        params={
            "text_encoder": params['text_encoder'],
            "vae": params['vae'],
            "unet": params['unet'],
            "safety_checker": params['safety_checker'],
        },
    )
else:
    print("Pipeline already saved to disk. Loading pipeline ...")
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        outdir,
        revision="flax",
        dtype=jnp.bfloat16,
    )

# %%
# Run inference with saved pipeline 

prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)

# parameters
p_params = replicate(params)
# arrays
prompt_ids = shard(prompt_ids)

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())

images = pipeline(prompt_ids, p_params, rng, jit=True)[0]


# %% 
# Show pipeline images

# convert numpy.ndarray to PIL.Image
def numpy_to_pil(image):
    image = image.squeeze(0) # remove batch dimension
    image = np.uint8(image * 255)
    # resize to 256x256
    image = Image.fromarray(image)
    image = image.resize((256, 256), Image.LANCZOS)
    return image

from diffusers.utils import make_image_grid
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
make_image_grid(images, rows=len(images)// jax.device_count(), cols=4)






# %%
#                                                        JAX inference (no pipeline)
 ###################################################################################

outdir = '/home/v/diffusers/examples/flax_models/stable-diffusion-v1-5'
revision = 'flax'
dtype = jax.numpy.bfloat16

# Load models and create wrapper for stable diffusion
tokenizer = CLIPTokenizer.from_pretrained(
    outdir,
    revision=revision,
    subfolder="tokenizer",
)
text_encoder = FlaxCLIPTextModel.from_pretrained(
    outdir, 
    revision=revision,
    subfolder="text_encoder",
)
vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    outdir,
    revision=revision,
    subfolder="vae",
    dtype=dtype,
)
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    outdir,
    revision=revision,
    subfolder="unet",
    dtype=dtype,
)
scheduler, scheduler_params = FlaxPNDMScheduler.from_pretrained(
    outdir,
    revision=revision,
    subfolder="scheduler",
)

# Create a params object
params = {
    "text_encoder": text_encoder.params,
    "vae": vae_params,
    "unet": unet_params,
    "scheduler": scheduler_params,
}


# %%
#                                                                 Inference function
####################################################################################

def my_generate(
    prompt_ids: jnp.array,
    params: Union[Dict, FrozenDict],
    prng_seed: jax.Array,
    num_inference_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    latents: Optional[jnp.ndarray] = None,
    neg_prompt_ids: Optional[jnp.ndarray] = None,
):
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

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    latents_shape = (
        batch_size,
        unet.config.in_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    if latents is None:
        latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
    else:
        if latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

    def loop_body(step, args):
        latents, scheduler_state = args
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        latents_input = jnp.concatenate([latents] * 2)

        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])

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

    scheduler_state = scheduler.set_timesteps(
        params["scheduler"], num_inference_steps=num_inference_steps, shape=latents.shape
    )

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * params["scheduler"].init_noise_sigma

    if DEBUG:
        # run with python for loop
        for i in range(num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        # latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))
        start_timestep_at = 0
        latents, _ = jax.lax.fori_loop(start_timestep_at, num_inference_steps, loop_body, (latents, scheduler_state))

    # scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.apply({"params": params["vae"]}, latents, method=vae.decode).sample

    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image



# %% 
#                       Prepare inputs and model state for parallelized inference
#################################################################################


prompts = 'A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, \
    40mm lens, shallow depth of field, close up, split lighting, cinematic'

neg_prompts = ''

# 2. We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

# Pmapped functions split inputs between devices, so we need to replicate the parameters
p_params = replicate(params)




def tokenize_inputs(prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]], tokenizer: CLIPTokenizer):
    if not isinstance(prompt, (str, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if not isinstance(negative_prompt, (str, list)):
        raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

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
    return text_input.input_ids, neg_text_input.input_ids


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng



# %% 
# Ahead of time compilation pmapped function

def aot_compile(
    prompt=prompts,
    neg_prompt=neg_prompts,
    seed=0,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=512,
    width=512,
    ):
    """
    Pmap distributes computation across all devices along the first axis.
    Any inputs that we don't want to split up should be duplicated first,
    so each that _generate receives a full copy of the input.

    After calling pmap, the prepared function p_generate will:
    - Make a copy of the underlying function, pipeline._generate, on each device.

    - Send each device a different portion of the input arguments (this is why 
      it's necessary to call the shard function).
      In this case, prompt_ids has shape (8, 1, 77, 768) so the array is 
      split into 8 and each copy of _generate receives an input with shape (1, 77, 768).

    The most important thing to pay attention to here is the batch size 
    (1 in this example), and the input dimensions that make sense for your code. 
    You don't have to change anything else to make the code work in parallel.
    
    NOTE: errors such as `pmap was requested to map its argument along axis 0,
    which implies that its rank should be at least 1, but is only 0?`
    are usually the result of not duplicate `params` or passing a scalar value 
    to `pmap` instead of an array.   
    """

    p, np = tokenize_inputs(prompt, neg_prompt, tokenizer) # tokenize inputs
    prompt_ids, neg_prompt_ids, prng_seed = replicate_all(p, np, seed) # duplicate inputs by x  NUM_DEVICES

    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]

    return (
        pmap(my_generate, static_broadcasted_argnums=[3, 4, 5]) # (5, 6, 7, 8) are the static args
        .lower(
            prompt_ids,
            p_params,
            prng_seed,
            num_inference_steps, # static (3)
            height,
            width,
            g,
            None, #latents,
            neg_prompt_ids,
            )
        .compile()
    )

    

    
# %% 
start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")

# %% 
def numpy_to_pil(images):
    if images.ndim == 3: images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def generate(prompt, negative_prompt, seed=0, guidance_scale=7.5):
    prompt_ids, neg_prompt_ids = tokenize_inputs(prompt, negative_prompt, tokenizer)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(prompt_ids, p_params, rng, g, None, neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return numpy_to_pil(np.array(images))


start = time.time()
neg_prompt = ""
images = generate(prompts, neg_prompts)
print(f"First inference in {time.time() - start}")

start = time.time()
images = generate(prompts, neg_prompts)
print(f"Inference in {time.time() - start}")

images[0].show()

# %% 
from diffusers.utils import make_image_grid
np_images = np.array(images)
pil_images = pipeline.numpy_to_pil(np_images)
make_image_grid(pil_images, rows=len(images)// jax.device_count(), cols=4)
# %%
