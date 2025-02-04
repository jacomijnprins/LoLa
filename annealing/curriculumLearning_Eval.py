import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

class SNLIEvalDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.labels = dataframe['label'].tolist()
        self.encodings = tokenizer(
            list(dataframe["premise"]),
            list(dataframe["hypothesis"]),
            truncation=True,
            padding='max_length',
            max_length=128
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def evaluate_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    # For reverse curriculum, change to "models/reverse_curriculum/reverse_curriculum_runs/final_model"
    model = RobertaForSequenceClassification.from_pretrained("models/curriculum_learning/curriculum_runs/final_model")
    tokenizer = RobertaTokenizer.from_pretrained("models/curriculum_learning/curriculum_runs/final_model")
    model.to(device)
    model.eval()

    # Load data
    val_df = pd.read_csv("data/raw/snli_val.csv")
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

    # Create a classification report string
    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=['entailment', 'neutral', 'contradiction']
    )
    print("\nClassification Report:")
    print(report)

    # Write accuracy and classification report to a txt file
    # For reverse curriculum, change to "results/CL_reversed_evaluation_report.txt"
    with open("results/evaluation_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)


if __name__ == "__main__":
    evaluate_model()



