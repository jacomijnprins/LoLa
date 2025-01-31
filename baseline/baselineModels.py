import torch
import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_checkpoint = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)


class CSVMetaDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        data = pd.read_csv(file_path)
        data = data.dropna(subset=['premise', 'hypothesis'])
        data = data[data['label'] != -1]
        self.texts = list(zip(data['premise'], data['hypothesis']))
        self.labels = data['label'].astype(int).tolist()
        self.encodings = tokenizer(
            list(data['premise']),
            list(data['hypothesis']),
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


# Training function
def train_model(metaset_subset_index):
    file_path = f"data/metaset_{metaset_subset_index}_subset.csv"
    print(f"Training on meta-dataset {metaset_subset_index} from file: {file_path}")

    # Create model
    model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    model.to(device)

    # Load dataset
    dataset = CSVMetaDataset(file_path, tokenizer)

    # Compute warm-up steps
    num_training_steps = len(dataset) // 16 * 3  # Batch size 16, epochs 3
    warmup_steps = int(0.05 * num_training_steps)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'models/meta_{metaset_subset_index}_results',
        eval_strategy='no',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        save_strategy='epoch',
        load_best_model_at_end=False,
        logging_dir=f'models/meta_{metaset_subset_index}_logs',
        logging_steps=10,
        lr_scheduler_type='linear',
        optim='adamw_torch',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model(f"models/meta_{metaset_subset_index}_model")
    print(f"Model for meta-dataset {metaset_subset_index} saved to models/meta_{metaset_subset_index}_model")


# train models one a time for computational efficiency
train_model(0)
