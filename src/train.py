import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

def set_seeds(seed_value=42):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

from model import SentimentBinaryClassifier
from data_loader import train_loader, test_loader, vocab_size

MODEL_SAVE_PATH = 'sentiment_model.pt'

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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.float().to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def start_training():    
    start_epoch = 0
    best_accuracy = 0.0

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading previous model from {MODEL_SAVE_PATH}...")
        checkpoint = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['accuracy']
        print(f"Resuming training from Epoch {start_epoch} with previous accuracy {best_accuracy:.4f}.")
    else:
        print("No saved model found. Starting training from scratch.")

    print("Starting model training...")
    print("-" * 30)

    num_epochs = 14

    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
            }, MODEL_SAVE_PATH)
            print(f"Model saved with accuracy {val_acc:.4f}.")

        print("-" * 30)

    print("Training finished.")

if __name__ == "__main__":
    start_training()