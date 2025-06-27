import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentBinaryClassifier
from data_loader import train_loader, test_loader, vocab_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentBinaryClassifier(vocab_size=vocab_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.float().to(device)
        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
