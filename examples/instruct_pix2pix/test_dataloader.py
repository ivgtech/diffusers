import unittest
import torch

# Assuming CustomTorchDataset is already defined as above
from jax_dataloader import CustomTorchDataset, NumpyLoader

class TestCustomTorchDataloader(unittest.TestCase):
    def setUp(self):
        # Create a sample data dictionary
        self.data = {
            'original_image': [torch.rand(3, 224, 224) for _ in range(10)],
            'edit_prompt': ['prompt' + str(i) for i in range(10)],
            'edited_image': [torch.rand(3, 224, 224) for _ in range(10)],
            'original_pixel_values': [torch.rand(3, 224, 224) for _ in range(10)],
            'edited_pixel_values': [torch.rand(3, 224, 224) for _ in range(10)],
            'input_ids': [torch.randint(0, 100, (10,)) for _ in range(10)]
        }
        self.dataset = CustomTorchDataset(self.data)
        self.batch_size = 3

    def test_dataloader_batch_size(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        for i, batch in enumerate(dataloader):
            expected_size = self.batch_size if i < len(dataloader) - 1 else len(self.dataset) % self.batch_size
            self.assertEqual(len(batch['edit_prompt']), expected_size)

    def test_dataloader_complete_data(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        total_batches = 0
        for batch in dataloader:
            total_batches += 1
            # Check if all keys are present in each batch
            for key in self.data.keys():
                self.assertIn(key, batch)
                # Check if each batch key has the correct batch size
                self.assertEqual(batch[key].shape[0], len(batch['edit_prompt']))
        # Check if total elements processed matches dataset size
        self.assertEqual(total_batches, (len(self.dataset) + self.batch_size - 1) // self.batch_size)

if __name__ == '__main__':
    unittest.main()