import os
import random
import torch
from torch.utils.data import DataLoader

def set_seeds(seed_value=42):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

from model import SentimentBinaryClassifier, MODEL_PATH
from data_loader import IMDBDataset, preprocess_text, collate_fn, words2idx, vocab_size, TEST_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentBinaryClassifier(vocab_size=vocab_size).to(device)

if os.path.exists(MODEL_PATH):
    print(f"Loading trained model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded with Val Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}.")
else:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train or download the model first.")
    exit()

model.eval()

class SpecificSentimentDataset(IMDBDataset):
    def __init__(self, data_dir, word2idx, sentiment_type):
        super().__init__(data_dir) 
        self.samples = []
        self.labels = []
        
        target_dir = os.path.join(data_dir, sentiment_type)
        expected_label = 1 if sentiment_type == 'pos' else 0

        if os.path.exists(target_dir):
            for filename in os.listdir(target_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(target_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        text = preprocess_text(text)
                        self.samples.append(text)
                        self.labels.append(expected_label)
        else:
            print(f"Warning: Directory {target_dir} not found for {sentiment_type} reviews.")

        self.word2idx = word2idx

if __name__ == "__main__":

    print("\n--- Predicting on positive IMDB reviews from test set ---")
    pos_test_dataset = SpecificSentimentDataset(TEST_DIR, words2idx, 'pos') 
    pos_test_loader = DataLoader(pos_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    correct_predictions_pos = 0
    total_predictions_pos = 0

    with torch.no_grad():
        for texts_indices, labels in pos_test_loader:
            texts_indices = texts_indices.to(device)
            labels = labels.float().to(device)

            outputs = model(texts_indices)
            preds = (outputs >= 0.5).float() 

            correct_predictions_pos += (preds == labels).sum().item()
            total_predictions_pos += labels.size(0)

    if total_predictions_pos > 0:
        accuracy_pos = correct_predictions_pos / total_predictions_pos
        print(f"\nTotal positive reviews predicted: {total_predictions_pos}")
        print(f"Correctly predicted as positive: {correct_predictions_pos}")
        print(f"Accuracy on positive reviews: {accuracy_pos:.4f} ({accuracy_pos*100:.2f}%)")
    else:
        print("No positive reviews found or processed in the test set 'pos' directory.")

    print("\n--- Predicting on negative reviews from test set ---")
    neg_test_dataset = SpecificSentimentDataset(TEST_DIR, words2idx, 'neg')
    neg_test_loader = DataLoader(neg_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    correct_predictions_neg = 0
    total_predictions_neg = 0

    with torch.no_grad():
        for texts_indices, labels in neg_test_loader:
            texts_indices = texts_indices.to(device)
            labels = labels.float().to(device)

            outputs = model(texts_indices)
            preds = (outputs >= 0.5).float()

            correct_predictions_neg += (preds == labels).sum().item()
            total_predictions_neg += labels.size(0)

    if total_predictions_neg > 0:
        accuracy_neg = correct_predictions_neg / total_predictions_neg
        print(f"\nTotal negative reviews predicted: {total_predictions_neg}")
        print(f"Correctly predicted as negative: {correct_predictions_neg}")
        print(f"Accuracy on negative reviews: {accuracy_neg:.4f} ({accuracy_neg*100:.2f}%)")
    else:
        print("No negative reviews found or processed in the test set 'neg' directory.")