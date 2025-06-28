import torch.nn as nn

class SentimentBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, output_dim=1, dropout_prob=0.5):
        super(SentimentBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return self.sigmoid(output).squeeze(1)