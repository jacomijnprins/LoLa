import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, default_data_collator
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")
else:
    print("CUDA is not available. Using CPU.")

model_paths = [
    'models/meta_0_model',
    'models/meta_1_model',
    'models/meta_2_model',
    'models/meta_3_model',
    'models/meta_4_model'
]

model_checkpoint = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

class CSVMetaDataset(Dataset):
    def __init__(self, file_path):
        print("loading dataset")
        data = pd.read_csv(file_path)
        print("dataset loaded")
        # Filter out invalid rows
        self.premises = data['premise'].tolist()
        self.hypotheses = data['hypothesis'].tolist()
        self.labels = data['label'].tolist()
        self.unique_ids = data['unique_id'].tolist()

        self.encodings = tokenizer(
            [f"{p} [SEP] {h}" for p, h in zip(self.premises, self.hypotheses)],
            truncation=True,
            padding=True,
            max_length=128
        )

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['unique_ids'] = self.unique_ids[idx]
        return item

# Paths to the meta-datasets
def evaluate_on_metaset(model, test_set, device):
    dataloader = DataLoader(test_set, batch_size=16, collate_fn=default_data_collator)
    model.eval()
    predictions, accuracies = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            predictions.extend(preds.cpu().tolist())
            accuracies.extend((preds == labels).float().cpu().tolist())

    return predictions, accuracies

# Evaluate one metaset
def evaluate_single_metaset(metaset_idx, metaset_file):
    print(f"Processing metaset {metaset_idx}...")

    # Load the metaset
    test_set = CSVMetaDataset(metaset_file)

    # Store results for the metaset
    results = pd.DataFrame({"unique_id": test_set.unique_ids, "true_label": test_set.labels})

    # Evaluate with all models except the one trained on this metaset
    for model_idx, model_path in enumerate(model_paths):
        if model_idx == metaset_idx:
            continue  # Skip the model trained on this metaset

        print(f"Evaluating metaset {metaset_idx} with model {model_idx}...")
        model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)

        predictions, accuracies = evaluate_on_metaset(model, test_set, device)

        # Add results for this model
        results[f"model_{model_idx}_predicted_label"] = predictions
        results[f"model_{model_idx}_accuracy"] = accuracies

    # Save the results for this metaset
    output_file = f"results/metasets_with_scores/difficulty_scores_metaset_{metaset_idx}.csv"
    results.to_csv(output_file, index=False)
    print(f"Saved results for metaset {metaset_idx} to {output_file}.")


metaset_idx = 4
metaset_file = f"data/metaset_{metaset_idx}.csv"
evaluate_single_metaset(metaset_idx, metaset_file)
