import os
import torch
import torch.nn as nn

MODEL_PATH = os.path.join('..', 'models', 'sentiment_model.pt')

class SentimentBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, output_dim=1, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.embedding(x.long())
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        output = torch.sigmoid(output).squeeze(1)
        return output