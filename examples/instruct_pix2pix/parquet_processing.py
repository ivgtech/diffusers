# %% 

import os
import jax
import numpy as np
from PIL import Image
import io
import pyarrow.parquet as pq
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer

# Assume the tokenizer is initialized here
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

class ParquetDataset(Dataset):
    def __init__(self, directory, transform=None, tokenizer=None, resolution=256):
        self.directory = directory
        self.transform = transform
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
        self.dataframes = [pq.read_table(fp).to_pandas() for fp in self.file_paths]
        self.original = []
        self.edited   = []
        self.prompts  = []
        self.prepare_data()

    def get_from_parquet(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes['bytes']))
        return image
        # image = image.convert("RGB").resize((self.resolution, self.resolution))
        # if self.transform:
        #     image = self.transform(image)
        
        # H,W,C = image.shape[-3:] # Get image dimensions as we're uncertain about the input shape (how many batch dimensions are present?)
        # return np.asarray(image.reshape(-1, H, W, C)) # Transpose dimensions to CHW format

    def prepare_data(self):
        original, edited, prompts = [], [], []

        for df in self.dataframes:
            # Assuming 'original_image' and 'edited_image' are columns with image bytes
            self.original += [self.get_from_parquet(img) for img in df['original_image']]
            self.edited   += [self.get_from_parquet(img) for img in df['edited_image']]
            self.prompts  += list(df['edit_prompt'])

        # self.original, self.edited, self.input_ids = self.preprocess_train(original, edited, prompts)



    def convert_to_np(self, image, resolution):
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)

    def preprocess_images(self, original, edited):

        original_image = self.convert_to_np(original, self.resolution)
        edited_image = self.convert_to_np(edited, self.resolution)

        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.

        images = np.concatenate([original_image, edited_image])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return self.transform(images) # NOTE: `transform` is a torchvision `Compose` object and so this method will return a torch tensor

    def preprocess_train(self, original, edited, prompt):

        # Preprocess images.
        preprocessed_images = self.preprocess_images(original, edited)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_image, edited_image = preprocessed_images.chunk(2)
        original_image = original_image.reshape(-1, 3, self.resolution, self.resolution)
        edited_image = edited_image.reshape(-1, 3, self.resolution, self.resolution)

        # Preprocess the captions.
        input_ids = self.tokenize_prompt(prompt)
        return original_image, edited_image, input_ids

    def tokenize_prompt(self, prompt):
        inputs = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids.squeeze()

    def __getitem__(self, idx):
        # np_original_pixel_values = np.stack([image for image in self.original]).astype(np.float32)
        # np_edited_pixel_values = np.stack([image for image in self.edited]).astype(np.float32)
        # np_input_ids = np.stack([prompt for prompt in self.input_ids])

        original_tensor, edited_tensor, prompts_tensor = self.preprocess_train(self.original[idx], self.edited[idx], self.prompts[idx])

        return {
            "original_pixel_values": original_tensor,
            "edited_pixel_values": edited_tensor,
            "input_ids": prompts_tensor,
        }


    def __len__(self):
        return len(self.prompts)


RESOLUTION = 256
CENTER_CROP = True
RANDOM_FLIP = True

train_transforms = transforms.Compose(
    [
        transforms.CenterCrop(RESOLUTION) if CENTER_CROP else transforms.RandomCrop(RESOLUTION),
        transforms.RandomHorizontalFlip() if RANDOM_FLIP else transforms.Lambda(lambda x: x),
    ]
)

def collate_fn_jax(examples):
    # Convert to numpy arrays and remove the singleton dimension (N,1,77 etc.)
    # np.stack applied to (M,1) or (1,M) -> (N,1,M)
    # np.stack applied to (M,) -> (N,M)
    # np.stack applied to (M,P) -> (N,M,P)
    np_original_pixel_values = np.stack([example["original_pixel_values"] for example in examples]).astype(np.float32).squeeze(1)
    np_edited_pixel_values = np.stack([example["edited_pixel_values"] for example in examples]).astype(np.float32).squeeze(1)
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



# Initialize dataset and DataLoader
dataset = ParquetDataset('data/', transform=train_transforms, tokenizer=tokenizer)

# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_jax, drop_last=True)

# # %% 
# # Example usage in a training loop
# for batch in dataloader:
#     images = batch['original_pixel_values']  # Images in CHW format ready for the model
#     input_ids = batch['input_ids']  # Tokenized captions
#     print(images.shape, images.dtype, images.max(), images.min())  
#     print(input_ids.shape)
#     break
#     # Further processing such as training goes here

# # # %%

# # %%
