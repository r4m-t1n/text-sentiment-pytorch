import torch
from model import SentimentBinaryClassifier, MODEL_PATH
from data_loader import preprocess_text, text_to_indices, words2idx, vocab_size

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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentimentBinaryClassifier(vocab_size=vocab_size).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    while True:
        text = input("Enter a text (or -1 to exit):\n")
        if text == "-1":
            break

        sentiment, probability = predict_text(model, text, words2idx, device)
        print(f"  Predicted Sentiment: {sentiment} (Probability: {probability:.4f})")