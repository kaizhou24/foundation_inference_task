from datasets import load_dataset

# GSM8K Data Loader for LLM
class DataLoaderGSM8K:
    def __init__(self):
        self.dataset = load_dataset("gsm8k", split="test")  # Loading test split

    def get_data(self, num_samples=10):
        return self.dataset.select(range(num_samples))

import os
from PIL import Image

class DataLoaderMME:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".jpg")]

    def get_data(self, num_samples=10):
        return self.data[:num_samples]  # Return first `num_samples` images
