import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm


class SNLIEvalDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = list(zip(dataframe['premise'], dataframe['hypothesis']))
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        premise, hypothesis = self.texts[idx]
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }


def evaluate_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained("models/curriculum_trained_model")
    tokenizer = RobertaTokenizer.from_pretrained("models/curriculum_trained_model")
    model.to(device)
    model.eval()

    # Load data
    val_df = pd.read_csv("data/snli_val.csv")  # Removed names parameter
    dataset = SNLIEvalDataset(val_df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch['labels'].tolist())

    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Print detailed results
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=['entailment', 'neutral', 'contradiction']
    ))


if __name__ == "__main__":
    evaluate_model()
