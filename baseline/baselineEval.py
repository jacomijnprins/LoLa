import torch
import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm


# Define evaluation dataset class
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


# Evaluation function
def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_df = pd.read_csv("../data/snli_val.csv")

    accuracies = []
    all_preds, all_labels = [], []

    for i in range(5):  # Iterate over all trained models
        print(f"Evaluating model meta_{i}...")
        model_path = f"models/meta_{i}_model"
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        dataset = SNLIEvalDataset(val_df, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        model_preds, model_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }

                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)

                model_preds.extend(preds.cpu().tolist())
                model_labels.extend(batch['labels'].tolist())

        accuracy = sum(p == l for p, l in zip(model_preds, model_labels)) / len(model_labels)
        print(f"Model meta_{i} Accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)

        all_preds.extend(model_preds)
        all_labels.extend(model_labels)

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")

    # Classification Report
    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=['entailment', 'neutral', 'contradiction']
    )
    print("\nClassification Report:")
    print(report)

    # Save results to log file
    os.makedirs("evaluation", exist_ok=True)
    log_file = "evaluation/evaluation_results.txt"
    with open(log_file, "w") as f:
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Evaluation results saved to {log_file}")


if __name__ == "__main__":
    evaluate_models()
