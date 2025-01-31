import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import os

class CSVDataset(Dataset):
    def __init__(self, file_path, tokenizer):
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
    

def train_model(dataset_name, model_checkpoint):
    if not os.path.exists('results'):
        os.makedirs('results')

    print(f'Training on curriculum from file: {dataset_name}')

    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

    dataset_path = f'curricula//{dataset_name}.csv'
    dataset = CSVDataset(dataset_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=f'results/results_{dataset_name}',
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
    trainer.save_model(f"results/{dataset_name}_model")



if __name__ == "__main__":
    dataset_name = 'most_ambiguous_balanced_5k' # When training different model only change dataset_name
    model_checkpoint = 'roberta-base'

    # check for gpu or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_model(dataset_name, model_checkpoint)