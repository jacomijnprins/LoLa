import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import os

model_checkpoint = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

class CSVSubsetDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.texts = list(zip(data['premise'], data['hypothesis']))
        self.labels = data['label'].astype(int).tolist()
        self.encodings = tokenizer(
            [f'{p} [SEP] {h}' for p, h in self.texts],
            truncation=True,
            padding=True,
            max_length=128
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

subsets_path = os.path.join("data", "snli_subsets")
subset_files = [os.path.join(subsets_path, f'subset_{i}.csv') for i in range(5)]


def train_model(subset_files):
    if not os.path.exists('results'):
        os.makedirs('results')

    for idx, file_path in enumerate(subset_files):
        print(f'Training on subset {idx + 1} from file: {file_path}')

        model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

        dataset = CSVSubsetDataset(file_path)

        training_args = TrainingArguments(
            output_dir=f'results/results_{idx + 1}',
            eval_strategy='no',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy='epoch',
            load_best_model_at_end=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(f"results/subset_{idx + 1}_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_model(subset_files)
