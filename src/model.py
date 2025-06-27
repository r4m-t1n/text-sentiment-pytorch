import torch.nn as nn
from data_loader import vocab_size

class SentimentBinaryClassifier(nn.Module):
    def __init__(self, vocab_size=vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1):
        super(SentimentBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return self.sigmoid(output).squeeze(1)