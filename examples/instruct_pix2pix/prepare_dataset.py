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


# Load the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer')

def tf_tokenize_captions(captions, max_length=77, tokenizer=tokenizer):
    # Wrap the tokenizer call in a tf.py_function to handle TensorFlow tensors
    def tokenize_fn(captions):
        # pad the captions to the max length
        input_ids = tokenizer(captions.numpy().decode('utf-8'), return_tensors="tf", padding='max_length', truncation=True, max_length=max_length)['input_ids']
        return input_ids[0] # NOTE: Avoids returning (B, 1, L) instead of (B, L)

    # Use tf.py_function to apply the tokenize_fn
    return tf.py_function(tokenize_fn, [captions], Tout=tf.int32)

# Function to decode image bytes and display the image
def decode_and_display_image(image_bytes):
    # Decode the image bytes to a tensor
    image = tf.image.decode_image(image_bytes, channels=3)
    # Convert the tensor to a numpy array for compatibility with PIL
    image = Image.fromarray(image.numpy())
    # Display the image
    image.show()

def tf_preprocess_images(images, resolution):
    # Function to resize and normalize images
    images = tf.image.resize(images, [resolution, resolution])
    images = (images / 127.5) - 1
    return images

def decode_and_process_images(image_bytes):
    # Decode JPEG images
    images = tf.image.decode_jpeg(image_bytes, channels=3)
    # Preprocess images (resizing and normalizing)
    return tf_preprocess_images(images, resolution=256)  # example resolution

def display_image(image):
    # Converts an implied JAX float array to a PIL Image
    # [-1.0, 1.0] -> [0.0, 255.0] -> [0, 255] -> jaxlib.xla_extension.DeviceList -> numpy.ndarray
    jax_impl_array = ((image + 1) * 127.5).astype(np.uint8) 
    image = Image.fromarray(np.array(jax_impl_array))      
    image.show()


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
        # return self.tf_dataset[idx]

        # Return a dictionary of the original image, edited image, and edit prompt
        return {
            "original_pixel_values": self.tf_dataset[idx][0].transpose(0, 3, 1, 2),
            "edited_pixel_values": self.tf_dataset[idx][1].transpose(0, 3, 1, 2),
            "input_ids": self.tf_dataset[idx][2],
        }
    
    def dataset_len(self):
        return len(self.tf_dataset)



def collate_fn_jax(batch):
    # Extract components from the batch
    orig_img_bytes, edit_img_bytes, input_ids = zip(*batch)

    # Convert to numpy arrays and TRANSPOSE to 
    original_images = [img.transpose(2,0,1) for img in orig_img_bytes]
    edited_images = [img.transpose(2,0,1) for img in edit_img_bytes]

    np_original_pixel_values = np.stack(original_images).astype(np.float32)
    np_edited_pixel_values = np.stack(edited_images).astype(np.float32)

    # Pad input_ids to the maximum length in the batch
    # max_length = max(len(ids) for ids in input_ids)
    # padded_input_ids = np.array([np.pad(ids, (0, max_length - len(ids)), mode='constant') for ids in input_ids])

    # input_ids = [ids[0] for ids in input_ids] # NOTE: Avoids returning (B, 1, L) instead of (B, L)
    padded_input_ids = np.stack(input_ids) # already padded by tokenizer
    
    
    # Convert numpy arrays to JAX arrays
    original_pixel_values = jax.device_put(np_original_pixel_values)
    edited_pixel_values = jax.device_put(np_edited_pixel_values)
    input_ids = jax.device_put(padded_input_ids)

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }



# Path to your parquet file
arquet_file = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'

# Create an IOTensor from the parquet file
dataset = tfio.IOTensor.from_parquet(parquet_file)

# Extract the image bytes and edit prompts from the dataset
original_image_bytes = dataset('original_image.bytes').to_tensor()
edited_image_bytes = dataset('edited_image.bytes').to_tensor()
edit_prompt_tensor = dataset('edit_prompt').to_tensor()

# Create TensorFlow datasets from these tensors
tf_original_images = tf.data.Dataset.from_tensor_slices(original_image_bytes)
tf_edited_images = tf.data.Dataset.from_tensor_slices(edited_image_bytes)
tf_edit_prompts = tf.data.Dataset.from_tensor_slices(edit_prompt_tensor)

# Zip the datasets together to create a single dataset
combined_dataset = tf.data.Dataset.zip((tf_original_images, tf_edited_images, tf_edit_prompts))

# Ensure that the dataset is evenly divisible by the number of devices
NUM_DEVICES = jax.device_count()


# Preprocess the combined dataset

processed_dataset = combined_dataset.map(

    lambda orig_img_bytes, edit_img_bytes, edit_prompt: (
        decode_and_process_images(orig_img_bytes),
        decode_and_process_images(edit_img_bytes),
        tf_tokenize_captions(edit_prompt)
    )
)

processed_dataset = processed_dataset.batch(batch_size=4, drop_remainder=True)
# Wrap the processed TensorFlow dataset
processed_dataset_wrapped = TFDatasetWrapper(processed_dataset)

tf_train_dataloader = processed_dataset_wrapped

# Creates a JAX-compatible DataLoader with the custom collate function
# tf_train_dataloader = DataLoader(
#     dataset=processed_dataset_wrapped,
#     batch_size=4,  # Adjust batch size as needed
#     shuffle=True,  # Enable shuffling if needed
#     collate_fn=collate_fn_jax  # Use the custom JAX-compatible collate function with padding
# )


# class NumpyLoader(data.DataLoader):
#     def __init__(
#         self,
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         sampler=None,
#         batch_sampler=None,
#         num_workers=0,
#         pin_memory=False,
#         drop_last=False,
#         timeout=0,
#         worker_init_fn=None,
#         collate_fn=None,
#         ):
#         super(self.__class__, self).__init__(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             batch_sampler=batch_sampler,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             timeout=timeout,
#             worker_init_fn=worker_init_fn,
#             collate_fn=collate_fn_jax
#             )

#         from datasets import IterableDataset
#         if isinstance(dataset, IterableDataset):
#             self.dataset_len =  self.dataset.info.splits['train'].num_examples
#         else:
#             self.dataset_len = len(self.dataset)

# tf_train_dataloader = NumpyLoader(
#     dataset=processed_dataset_wrapped,
#     batch_size=4,  # Adjust batch size as needed
#     shuffle=True,  # Enable shuffling if needed
#     collate_fn=collate_fn_jax  # Use the custom JAX-compatible collate function with padding
# )



# %%

# # Print the total number of examples

def fn_wrapper():
    print(f"total # examples ({len(tf_train_dataloader) * 4}) = #batches({len(tf_train_dataloader)}) * batch_size({4})")

    # Example usage: Iterate over the DataLoader
    for batch in tf_train_dataloader:
        if isinstance(batch, tuple):
            batch = batch[0]

            for orig in batch:
                print(orig.shape)
                display_image(orig )
                print((batch).shape)
            break
        # print("Batch keys:", batch.keys())
        print("batch shape:", batch["input_ids"].shape)
        orig = batch["original_pixel_values"][0]
        prompt = batch["input_ids"][0]
        edited = batch["edited_pixel_values"][0]

        display_image(orig.transpose(1, 2, 0) )
        print(tokenizer.batch_decode([prompt], skip_special_tokens=True))

        print("Edited Image Pixel Values Shape:", edited.shape)
        print("Input IDs Shape:", prompt.shape)
        break

# fn_wrapper()
# %%

