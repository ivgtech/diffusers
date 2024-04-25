# %%
import jax 
import jax.numpy as jnp
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_io as tfio
from transformers import CLIPTokenizer
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

# Function to decode image bytes and display the image
def decode_and_display_image(image_bytes):
    # Decode the image bytes to a tensor
    image = tf.image.decode_image(image_bytes, channels=3)
    # Convert the tensor to a numpy array for compatibility with PIL
    image = Image.fromarray(image.numpy())
    # Display the image
    image.show()

# Load the tokenizer

tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer')

def tokenize_captions(captions, max_length=77, tokenizer=tokenizer):
    # Wrap the tokenizer call in a tf.py_function to handle TensorFlow tensors
    def tokenize_fn(captions):
        # pad the captions to the max length
        return tokenizer(captions.numpy().decode('utf-8'), return_tensors="tf", padding='max_length', truncation=True, max_length=max_length)['input_ids']
    # Use tf.py_function to apply the tokenize_fn
    return tf.py_function(tokenize_fn, [captions], Tout=tf.int32)



# Path to your parquet file
parquet_file = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'
# Create an IOTensor from the parquet file
dataset = tfio.IOTensor.from_parquet(parquet_file)

original_image_bytes = dataset('original_image.bytes').to_tensor()
edited_image_bytes = dataset('edited_image.bytes').to_tensor()
edit_prompt_tensor = dataset('edit_prompt').to_tensor()

# Create TensorFlow datasets from these tensors
tf_original_images = tf.data.Dataset.from_tensor_slices(original_image_bytes)
tf_edited_images = tf.data.Dataset.from_tensor_slices(edited_image_bytes)
tf_edit_prompts = tf.data.Dataset.from_tensor_slices(edit_prompt_tensor)

# Zip the datasets together to create a single dataset
combined_dataset = tf.data.Dataset.zip((tf_original_images, tf_edited_images, tf_edit_prompts))

# batch the combined dataset
# combined_dataset = combined_dataset.batch(4)
# shuffle the combined dataset
# combined_dataset = combined_dataset.shuffle(100)

# Assuming 'column_name' is the name of one of the columns in your parquet file
# column_name = 'original_image.bytes'
# column_dataset = dataset(column_name).to_tensor()

# Now create a TensorFlow dataset from this tensor
# tf_dataset = tf.data.Dataset.from_tensor_slices(column_dataset)
# tf_dataset = tf_dataset.batch(2)  # Example of batching
# tf_dataset = tf_dataset.shuffle(100)  # Example of shuffling


# <ParquetIOTensor: spec=(TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'original_prompt'>),
#   TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'original_image.bytes'>), 
#   TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'original_image.path'>), 
#   TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'edit_prompt'>), 
#   TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'edited_prompt'>), 
#   TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'edited_image.bytes'>), 
#   TensorSpec(shape=(1195,), dtype=tf.string, name=<tf.Tensor: shape=(), dtype=string, numpy=b'edited_image.path'>))>





# Assuming 'tf_edit_prompts' is a TensorFlow dataset containing text prompts
# tf_edit_prompts = tf.data.Dataset.from_tensor_slices(["example prompt"])
# tokenized_prompts = tf_edit_prompts.map(tokenize_captions)

# # Example usage: Iterate over the tokenized prompts dataset
# for tokenized_prompt in tokenized_prompts.take(1):
#     print("Tokenized Prompt:", tokenized_prompt)


def preprocess_images(images, resolution):
    # Function to resize and normalize images
    images = tf.image.resize(images, [resolution, resolution])
    images = (images / 127.5) - 1
    return images

def decode_and_process_images(image_bytes):
    # Decode JPEG images
    images = tf.image.decode_jpeg(image_bytes, channels=3)
    # Preprocess images (resizing and normalizing)
    return preprocess_images(images, resolution=256)  # example resolution

processed_dataset = combined_dataset.map(
    lambda orig_img_bytes, edit_img_bytes, edit_prompt: (
        decode_and_process_images(orig_img_bytes),
        decode_and_process_images(edit_img_bytes),
        tokenize_captions(edit_prompt)
    )
)


class TFDatasetWrapper(Dataset):
    """ A PyTorch Dataset wrapper for TensorFlow datasets. """
    def __init__(self, tf_dataset):
        """ Convert TensorFlow dataset to a list to make it subscriptable. """
        self.tf_dataset = list(tf_dataset.as_numpy_iterator())

    def __len__(self):
        """ Return the number of items in the dataset. """
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        """ Retrieve an item by index. """
        return self.tf_dataset[idx]


def collate_fn_jax(batch):
    # Extract components from the batch
    orig_img_bytes, edit_img_bytes, input_ids = zip(*batch)

    # Convert to numpy arrays
    np_original_pixel_values = np.stack(orig_img_bytes).astype(np.float32)
    np_edited_pixel_values = np.stack(edit_img_bytes).astype(np.float32)

    # Pad input_ids to the maximum length in the batch
    max_length = max(len(ids) for ids in input_ids)
    padded_input_ids = np.array([np.pad(ids, (0, max_length - len(ids)), mode='constant') for ids in input_ids])

    # Convert numpy arrays to JAX arrays
    original_pixel_values = jax.device_put(np_original_pixel_values)
    edited_pixel_values = jax.device_put(np_edited_pixel_values)
    input_ids = jax.device_put(padded_input_ids)

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        #"input_ids": input_ids,
    }


# Wrap the processed TensorFlow dataset
processed_dataset_wrapped = TFDatasetWrapper(processed_dataset)

# Assuming processed_dataset_wrapped is already defined
train_dataloader = DataLoader(
    dataset=processed_dataset_wrapped,
    batch_size=4,  # Adjust batch size as needed
    shuffle=True,  # Enable shuffling if needed
    collate_fn=collate_fn_jax  # Use the custom JAX-compatible collate function with padding
)

# %% 
# Example usage: Iterate over the DataLoader
for batch in train_dataloader:
    print("Batch keys:", batch.keys())
    print("Original Image Pixel Values Shape:", batch["original_pixel_values"].shape)
    print("Edited Image Pixel Values Shape:", batch["edited_pixel_values"].shape)
    #print("Input IDs Shape:", batch["input_ids"].shape)





# %%


# %%

# Example usage: Iterate over the combined dataset
# for orig_img_bytes, edit_img_bytes, edit_prompt in combined_dataset.take(1):
#     print(edit_prompt.numpy().shape)
#     for i, bp in enumerate(edit_prompt.numpy()):
#         print(bp.decode('utf-8'))
#         decode_and_display_image(orig_img_bytes[i].numpy())
#         decode_and_display_image(edit_img_bytes[i].numpy())

    # for batch in edit_prompt.numpy():
    #     print(batch.decode('utf-8'))
    # for batch in orig_img_bytes.numpy():
    #     decode_and_display_image(batch)
    # for batch in edit_img_bytes.numpy():
    #    decode_and_display_image(batch)
#     print("Edit Prompt:", edit_prompt.numpy().decode('utf-8'))
#     # Decode and display images if needed
#     decode_and_display_image(orig_img_bytes)
#     decode_and_display_image(edit_img_bytes)



# # %%

# %%

for orig_img_bytes, edit_img_bytes, edit_prompt in processed_dataset.take(1):
    # Convert to numpy arrays for JAX compatibility
    orig_img_bytes = orig_img_bytes.numpy()
    edit_img_bytes = edit_img_bytes.numpy()
    edit_prompt = edit_prompt.numpy()
    print("Original Image Shape:", type(orig_img_bytes) , orig_img_bytes.shape, orig_img_bytes)
    print("Edited Image Shape:", edit_img_bytes.shape)
    print("Tokenized Prompt Shape:", edit_prompt.shape)

# Data loaders 
def collate_fn_jax(batch):
    orig_img_bytes, edit_img_bytes, input_ids = zip(*batch)
    examples = {}
    examples['original_pixel_values'] = np.array(orig_img_bytes)
    examples['edited_pixel_values'] = np.array(edit_img_bytes)
    examples['input_ids'] = np.array(input_ids)
    

    np_original_pixel_values = np.stack([example["original_pixel_values"] for example in examples]).astype(np.float32)
    np_edited_pixel_values = np.stack([example["edited_pixel_values"] for example in examples]).astype(np.float32)
    np_input_ids = np.stack([example["input_ids"] for example in examples])

    # Move data to accelerator
    original_pixel_values  = jax.device_put(np_original_pixel_values)
    edited_pixel_values = jax.device_put(np_edited_pixel_values)
    input_ids = jax.device_put(np_input_ids)

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        collate_fn=None,
        ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_jax
            )

        from datasets import IterableDataset
        if isinstance(dataset, IterableDataset):
            self.dataset_len =  self.dataset.info.splits['train'].num_examples
        else:
            self.dataset_len = len(self.dataset)
            

# %%
# %%