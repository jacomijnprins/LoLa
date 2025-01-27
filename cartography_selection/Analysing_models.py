import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset


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


def log_results(log_file, model_name, metrics):
    with open(log_file, 'a') as f:
        f.write(f'Model {model_name}:\n')
        f.write(f'model accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'model precision: {metrics["precision"]:.4f}\n')
        f.write(f'model recall: {metrics["recall"]:.4f}\n')
        f.write(f'model f1: {metrics["f1"]:.4f}\n')
        f.write("\n")


if __name__ == "__main__":
    model_name = "most_ambiguous" # only change this for each model

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    val_data = pd.read_csv('data/snli_val.csv')
    val_dataset =SNLIDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    model_path = f'results//{model_name}_model'
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    accuracy, precision, recall, f1 = eval_model(model, val_loader)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    log_results('results/model_evaluation_log.txt', model_name, metrics)

    print(f'accuracy: {metrics["accuracy"]:.4f}, '
        f'precision: {metrics["precision"]:.4f}, '
        f'recall: {metrics["recall"]:.4f}, '
        f'f1: {metrics["f1"]:.4f}')