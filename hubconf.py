import os
import pickle
import torch
from src.model import SentimentBinaryClassifier
from src.data_loader import preprocess_text, text_to_indices
from src.predict import predict_text

dependencies = ['numpy==1.24.3', 'scikit-learn==1.4.2', 'torch==2.3.0']

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pt')
WORDS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'words2idx.pickle')

def _load_vocab():
    if not os.path.exists(WORDS_PATH):
        raise FileNotFoundError(f"Vocabulary file not found at {WORDS_PATH}. ")
    with open(WORDS_PATH, 'rb') as f:
        words2idx = pickle.load(f)
    return words2idx

def sentiment_classifier(pretrained=True, device=None, **kwargs):
    words2idx = _load_vocab()
    vocab_size = len(words2idx)

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

predict = predict_text
preprocess = preprocess_text
text2idx = text_to_indices