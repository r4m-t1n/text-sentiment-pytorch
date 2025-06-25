import os
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class IMDBDataset(Dataset):
    def __init__(self, data_dir):
        super(IMDBDataset, self).__init__()

        self.data_dir = data_dir
        self.samples = []
        self.labels = []

        pos_dir = os.path.join(self.data_dir, 'pos')
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.samples.append(text)
                    self.labels.append(1)

        neg_dir = os.path.join(self.data_dir, 'neg')
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.samples.append(text)
                    self.labels.append(0)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

