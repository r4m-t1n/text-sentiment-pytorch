import os
import re
import pickle
import torch
from src.model import SentimentBinaryClassifier

dependencies = ['numpy==1.24.3', 'scikit-learn==1.4.2', 'torch==2.3.0']

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pt')
WORDS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'words2idx.pickle')

def _load_vocab():
    if not os.path.exists(WORDS_PATH):
        raise FileNotFoundError(f"Vocabulary file not found at {WORDS_PATH}. ")
    with open(WORDS_PATH, 'rb') as f:
        words2idx = pickle.load(f)
    return words2idx

words2idx = _load_vocab()
vocab_size = len(words2idx)

def sentiment_classifier(pretrained=True, device=None, **kwargs):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentimentBinaryClassifier(vocab_size=vocab_size, **kwargs).to(device)

    if pretrained:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Pre-trained model not found at {MODEL_PATH}. "
                                    "Please train the model using train.py")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model with Val Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}.")

    return model

def predict_text(model, text, word2idx, device):

    processed_text = preprocess_text(text)

    indices = text_to_indices(processed_text, word2idx)
    
    if not indices:
        print("Warning: Input text became empty after preprocessing.")
        return "Empty", 0.5

    text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(text_tensor)

    probability = output.item()

    prediction = (probability >= 0.5)
    sentiment = "(+)Positive" if prediction else "(-)Negative"
    
    return sentiment, probability

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