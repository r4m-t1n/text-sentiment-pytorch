import os
import re
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_files

DATA_DIR = os.path.join('..', 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'aclImdb', 'train')
TEST_DIR = os.path.join(DATA_DIR, 'aclImdb', 'test')

class IMDBDataset(Dataset):
    def __init__(self, data_dir, word2idx=None):
        super().__init__()

        self.data_dir = data_dir

        imdb_data = load_files(
            data_dir, 
            shuffle=True, 
            random_state=42, 
            encoding='utf-8'
        )

        raw_samples = imdb_data.data
        self.labels = imdb_data.target.tolist() 

        self.samples = [preprocess_text(text) for text in raw_samples]

        self.word2idx = word2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]

        if self.word2idx:
            text = text_to_indices(text, self.word2idx)

        return text, label

    @staticmethod
    def build_vocab(texts: list, max_vocab_size=30000, min_frequency=2):
        counter = Counter()
        for text in texts:
            words = text.split()
            counter.update(words)
        vocab = {word for word, freq in counter.most_common(max_vocab_size) if freq >= min_frequency}
        vocab = list(vocab)

        word2idx = {word: idx+2 for idx, word in enumerate(vocab)}
        word2idx['<PAD>'] = 0
        word2idx['<UNK>'] = 1
        return word2idx

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?;:]', '', text) 
    return text.strip()

def text_to_indices(text, word2idx):
    indices = []
    for word in text.split():
        if word in word2idx:
            indices.append(word2idx[word])
        else:
            indices.append(word2idx['<UNK>'])
    return indices

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = [torch.tensor(text, dtype=torch.long) for text in texts]
    labels = torch.tensor(labels, dtype=torch.long)

    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return padded_texts, labels

print('Importing data_loader...')

print('Loading temp_train...')
temp_train = IMDBDataset(TRAIN_DIR) 
print('temp_train loaded.')

print('Building vocabulary...')
words2idx = IMDBDataset.build_vocab(temp_train.samples)
vocab_size = len(words2idx)
print(f'Vocabulary is built. Size: {vocab_size}')

print('Defining train dataset...')
train = IMDBDataset(TRAIN_DIR, words2idx)
print('Train dataset defined.\nDefining test dataset...')
test = IMDBDataset(TEST_DIR, words2idx)
print('Test dataset defined.')

print('Loading train dataset...')
train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_fn)
print('Train dataset loaded.\nLoading test dataset...')
test_loader = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collate_fn)
print('test dataset loaded.')

print('data_loader imported successfully.')