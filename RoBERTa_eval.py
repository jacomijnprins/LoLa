import os
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset

models_dir = "./results"
val_path = './data/snli_val.csv'
num_models = 5
log_file = './results/evaluation_log.txt'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class SNLIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length = 128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
            return len(self.dataframe)

    def __getitem__(self, idx):
        premise = self.dataframe.iloc[idx]['premise']
        hypothesis = self.dataframe.iloc[idx]['hypothesis']
        label = self.dataframe.iloc[idx]['label']

        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

val_data = pd.read_csv(val_path)
val_dataset =SNLIDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def eval_model(model, data_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

def log_results(model_idx, metrics):
    with open(log_file, 'a') as f:
        f.write(f'model {model_idx}:\n')
        f.write(f'model accuracy: {metrics["accuracy"]:.4f}:\n')
        f.write(f'model precision: {metrics["precision"]:.4f}:\n')
        f.write(f'model recall: {metrics["recall"]:.4f}:\n')
        f.write(f'model f1: {metrics["f1"]:.4f}:\n')
        f.write("\n")

all_metrics = []

for i in range(num_models):
    model_path = os.path.join(models_dir, f'subset_{i+1}_model')
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    accuracy, precision, recall, f1 = eval_model(model, val_loader)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    all_metrics.append(metrics)

    log_results(i+1, metrics)
    print(f'model {i+1} - accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1 score: {f1:.4f}')

average_metrics = {
    'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
    'precision': np.mean([m['precision'] for m in all_metrics]),
    'recall': np.mean([m['recall'] for m in all_metrics]),
    'f1': np.mean([m['f1'] for m in all_metrics])
}

with open(log_file, 'a') as f:
    f.write('Average Metrics:\n')
    f.write(f'accuracy: {average_metrics["accuracy"]:.4f}\n')
    f.write(f'precision: {average_metrics["precision"]:.4f}\n')
    f.write(f'recall: {average_metrics["recall"]:.4f}\n')
    f.write(f'f1: {average_metrics["f1"]:.4f}\n')

print(f'average metrics - accuracy: {average_metrics["accuracy"]:.4f}, '
      f'precision: {average_metrics["precision"]:.4f}, '
      f'recall: {average_metrics["recall"]:.4f}, '
      f'f1: {average_metrics["f1"]:.4f}')





