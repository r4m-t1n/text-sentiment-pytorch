import os
import re
import random
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join('..', 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'aclimdb', 'train')
TEST_DIR = os.path.join(DATA_DIR, 'aclimdb', 'test')

class IMDBDataset(Dataset):
    def __init__(self, data_dir, size: int, word2idx=None):
        super(IMDBDataset, self).__init__()

        self.data_dir = data_dir
        self.samples = []
        self.labels = []

        random.seed(42)

        pos_dir = os.path.join(self.data_dir, 'pos')
        pos_files = random.sample(os.listdir(pos_dir), min(size, len(os.listdir(pos_dir))))
        for filename in pos_files:
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = self._preprocess_text(text)
                    self.samples.append(text)
                    self.labels.append(1)

        neg_dir = os.path.join(self.data_dir, 'neg')
        neg_files = random.sample(os.listdir(neg_dir), min(size, len(os.listdir(neg_dir))))
        for filename in neg_files:
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = self._preprocess_text(text)
                    self.samples.append(text)
                    self.labels.append(0)
        
        self.word2idx = word2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]

        if self.word2idx:
            text = self._text_to_indices(text, self.word2idx)

        return text, label

    def _preprocess_text(self, text):
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,!?;:]', '', text) 
        return text.strip()

    
    def _text_to_indices(self, text, word2idx):
        indices = []
        for word in text.split():
            if word in word2idx:
                indices.append(word2idx[word])
            else:
                indices.append(word2idx['<UNK>'])
        return indices

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

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = [torch.tensor(text) for text in texts]
    labels = torch.tensor(labels)

    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return padded_texts, labels

temp_train = IMDBDataset(TRAIN_DIR, 12500) 

words2idx = IMDBDataset.build_vocab(temp_train.samples)
vocab_size = len(words2idx)

train = IMDBDataset(TRAIN_DIR, 12500, words2idx)
test = IMDBDataset(TEST_DIR, 12500, words2idx)

train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collate_fn)