import unittest
import torch

from jax_dataloader import CustomTorchDataset, NumpyLoader

# Assuming CustomTorchDataset is already defined as above

class TestCustomTorchDataset(unittest.TestCase):
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

    def test_length(self):
        # Test that the length method returns the correct number of items
        self.assertEqual(len(self.dataset), 10)

    def test_getitem(self):
        # Test that __getitem__ returns the correct item
        item = self.dataset[0]
        self.assertTrue('original_image' in item)
        self.assertTrue('edit_prompt' in item)
        self.assertTrue('edited_image' in item)
        self.assertTrue('original_pixel_values' in item)
        self.assertTrue('edited_pixel_values' in item)
        self.assertTrue('input_ids' in item)
        self.assertEqual(item['edit_prompt'], 'prompt0')

    def test_index_error(self):
        # Test that index errors are raised when expected
        with self.assertRaises(IndexError):
            self.dataset[10]  # Out of range

    def test_dataloader(self):
        # Test iterating through a dataloader
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=2)
        for batch in dataloader:
            self.assertEqual(len(batch['edit_prompt']), 2)  # Check batch size
            break  # Only test one batch for simplicity





in__':

if __name__ == '__main__':
    unittest.main()