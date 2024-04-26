#  %%
from args import *

# curl -X GET \
#      "https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train"


# URL to fetch the parquet files list
url = "https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train"

# Perform a GET request to retrieve the data
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    parquet_files = response.json()
    print("List of Parquet files:", parquet_files)
else:
    print("Failed to retrieve Parquet files. Status code:", response.status_code)

# select the first 10 parquet files
parquet_files = parquet_files[:10]



# *** Create JAX dataset from parquet file ***

parquet_file = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'
tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer', dtype=jnp.bfloat16)
LEN_TRAIN_DATASET = pq.read_table(parquet_file).num_rows

def create_dataset_from_parquet(parquet_file, batch_size):
    """Creates a JAX-compatible dataset from a Parquet file with preprocessing."""
    table = pq.read_table(parquet_file)
    DATASET_SIZE = table.num_rows
    data = table.to_pydict()

    # Assuming columns: "original_image", "edit_prompt", "edited_image"
    original_images = data["original_image"]
    edit_prompts = data["edit_prompt"]
    edited_images = data["edited_image"]

    # Preprocessing steps
    def preprocess_image(image_bytes):
      path = image_bytes['path']
      bytes = image_bytes['bytes']
      """Preprocesses a single image."""
      image = tf.image.decode_jpeg(bytes, channels=3)

      image = tf.image.resize(image, [256, 256])
      image = tf.cast(image, tf.float32) / 255.0

      # Transpose the image to [H, W, C] format
      image = tf.transpose(image, [2, 0, 1])

      return image

    # def preprocess_prompt(prompt):
    #   """Preprocesses a single text prompt."""
    #   # Tokenize the prompt using your defined tokenizer
    #   input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="np").input_ids
    #   input_ids = input_ids.squeeze(0)  # Remove the batch dimension as stacking (1,77) arrays creates (N, 1, 77) shape and we expect (N, 77)
    #   return input_ids

    # # Apply preprocessing
    # original_images = np.stack([preprocess_image(img) for img in original_images])
    # edited_images = np.stack([preprocess_image(img) for img in edited_images])
    # edit_prompts = np.stack([preprocess_prompt(prompt) for prompt in edit_prompts])

    # Create dictionary of preprocessed data
    preprocessed_data = {
        "original_pixel_values": original_images,
        "input_ids": edit_prompts,
        "edited_pixel_values": edited_images,
    }

    return preprocessed_data

    # Split into batches and convert to JAX arrays
    num_samples = len(original_images)
    num_batches = num_samples // batch_size
    batched_data = {}
    for key, value in preprocessed_data.items():
      batched_data[key] = np.array_split(value[:num_batches * batch_size], num_batches) # np.array_split returns a list
    
    return jax.tree_util.tree_map(jnp.array, batched_data)

# def create_input_pipeline(batch_size, parquet_file):
#   """Creates the distributed data loading pipeline with manual sharding."""
#   dataset = create_dataset_from_parquet(parquet_file, batch_size)

#   # Manual sharding with jax.lax.dynamic_slice
#   def shard_fn(x):
#     local_device_count = jax.local_device_count()
#     return jax.lax.dynamic_slice(
#         x, (jax.process_index() * local_device_count,) + (0,) * (x.ndim - 1),
#         (local_device_count,) + x.shape[1:])

#   sharded_dataset = jax.tree_util.tree_map(shard_fn, dataset)
#   return sharded_dataset

# Example usage in your training loop:
# per_device_batch_size = args.train_batch_size  
# train_dataset = create_input_pipeline(per_device_batch_size, parquet_file)


train_dataset = create_dataset_from_parquet(parquet_file, args.train_batch_size)

# %%
# Create  an iterator to loop through the dataset

def create_iterator(dataset):
  keys = list(dataset.keys())
  for vals in zip(*dataset.values()):
    yield {key: val for key, val in zip(keys, vals)}


# train_dataloader = create_iterator(train_dataset)

# %%

# for i, batch in enumerate(zip(*train_dataset.values())):
# for i,  batch in enumerate(train_dataloader):
#   print(batch.keys())
#   break
#   # ... (your training step logic using original_images, input_ids, edited_images) ...

#   #  for i, batch in enumerate(train_dataset.values()):
#   original_images, input_ids, edited_images = batch.values()
#   print(original_images.shape, input_ids.shape, edited_images.shape)
#   #print(type(batch), len(batch), len(batch[0]))
#   #print((batch[1].shape))
#   # ... (your training step logic using original_images, input_ids, edited_images) ...
# print(i)

  
  
  




# # %%
