import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import requests
import numpy as np

from config import ROOT_DIR


class AliceInWonderlandDataset(Dataset):
    file_url = "https://www.gutenberg.org/cache/epub/35688/pg35688.txt"
    raw_data_path = os.path.join(ROOT_DIR, '.rawdata', 'alice_in_wonderland.raw.txt')
    tokenizer_path = os.path.join(ROOT_DIR, '.rawdata', 'alice_in_wonderland.tokenizer.bpe.json')
    data = ""
    seq_length = 5

    def __init__(self, train, download=False, tokenizer=None):
        self.train = train
        self.tokenizer = tokenizer

        if not os.path.exists(os.path.join(ROOT_DIR, '.rawdata')):
            os.mkdir(os.path.join(ROOT_DIR, '.rawdata'))


        if not os.path.exists(self.raw_data_path) or download:
            self._download_data(self.file_url, self.raw_data_path)
        self._load_data()

    def _download_data(self, url, file_path):
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    def _load_data(self):
        with open(self.raw_data_path, "r", encoding="utf-8") as file:
            self.data = file.read()

        self.data = self.data[:self.data.find("End of Project Gutenberg's")]
        self.data = self.data[self.data.find("_ALICE'S home. LEWIS CARROLL is discovered, playing chess."):]

        # Clean raw data
        self.data = self.data.lower()
        self.data = self.data.replace('_', '')
        self.data = self.data.replace('--', ' ')
        self.data = self.data.replace('-', ' ')
        self.data = self.data.replace('\n', ' ')

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

            trainer = trainers.BpeTrainer(vocab_size=100)
            self.tokenizer.train_from_iterator(self.data.splitlines(), trainer=trainer)
            self.tokenizer.save(self.tokenizer_path)


        tokens = self.tokenizer.encode(self.data).ids
        X_ids, Y_ids = [], []

        for i in range(len(tokens) - self.seq_length):
            X_ids.append(tokens[i:i + self.seq_length])
            Y_ids.append(tokens[i + self.seq_length])

        all_X = np.array(X_ids)
        all_Y = np.array(Y_ids)

        indices = np.arange(len(all_X))

        # Hardcoded seed, slightly hacky way of allowing train=True/False to split at the same point
        np.random.seed(54321)
        np.random.shuffle(indices)
        split = len(indices) // 2

        if self.train:
            indices = indices[:split]
        else:
            indices = indices[split:]

        self.X = torch.tensor(all_X[indices], dtype=torch.long)
        self.Y = torch.tensor(all_Y[indices], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

if __name__ == '__main__':
    dataset_train = AliceInWonderlandDataset(train=True)