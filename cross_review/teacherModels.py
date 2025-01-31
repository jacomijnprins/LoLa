import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

# Model and tokenizer setup
model_checkpoint = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

# Custom Dataset Class
class CSVMetaDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        # Filter out invalid rows
        data = data.dropna(subset=['premise', 'hypothesis'])
        data = data[data['label'] != -1]
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

# Paths to meta-dataset files
meta_path = "../data"
meta_files = [os.path.join(meta_path, f'processed/metaset_{i}.csv') for i in range(5)]

# Training function
def train_models(meta_files):
    # Create results directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    for idx, file_path in enumerate(meta_files):
        print(f"Training on meta-dataset {idx} from file: {file_path}")

        # Initialize model
        model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

        # Load dataset
        dataset = CSVMetaDataset(file_path)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'models/meta_{idx}_results',  # Save model and logs here
            eval_strategy='no',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            save_strategy='epoch',
            load_best_model_at_end=False,
            logging_dir=f'models/meta_{idx}_logs',  # TensorBoard logs
            logging_steps=10,
            lr_scheduler_type='linear',
            optimizer='adamw_torch'
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
        trainer.save_model(f"models/meta_{idx}_model")

        print(f"Model for meta-dataset {idx} saved to models/meta_{idx}_model")

# Device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Train models on meta-datasets
train_models(meta_files)
